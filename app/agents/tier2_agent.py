"""
Tier 2 agent — uses Gemini API for full multi-step reasoning.

Rules:
- Never directly calls document tools.
- Only outputs structured AgentAction via provider.
- Submits actions to Harness.
- Continues from existing AgentState (supports escalation handoff).
"""
import logging
from typing import List, Dict

from app.schemas.state import AgentState, AgentStatus, Tier
from app.schemas.responses import FinalAnswer, VerificationStatus
from app.harness.harness import Harness
from app.providers.base import LLMProvider
from app.verification.verifier import DeterministicVerifier

_log = logging.getLogger("rap.tier2")

MAX_ITERATIONS = 10  # Hard ceiling to prevent infinite loops


class Tier2Agent:
    def __init__(self, harness: Harness, provider: LLMProvider):
        self.harness = harness
        self.provider = provider
        self.det_verifier = DeterministicVerifier()

    def run(self, state: AgentState, doc_metadata: List[Dict]) -> FinalAnswer:
        state = state.model_copy(deep=True)
        state.current_tier = Tier.TIER_2

        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1

            # Step 1: Check budget
            if state.calls_remaining == 0:
                _log.info("Budget exhausted — entering verification.")
                state.status = AgentStatus.BUDGET_EXHAUSTED
                break

            # Step 2: Run deterministic pre-check
            det_result = self.det_verifier.check(state)
            if det_result.status == VerificationStatus.INSUFFICIENT and state.calls_used > 0:
                # If deterministic check says insufficient AND we have contradictions — stop early
                if state.contradictions:
                    _log.info("Deterministic verifier found unresolved contradictions — stopping.")
                    break

            # Step 3: Ask Gemini for next action
            action = self.provider.generate_tier2_action(state, doc_metadata)
            _log.info(f"Tier2 action: {action.action!r} args={action.arguments} reason={action.reason!r}")

            if action.action == "STOP":
                _log.info("Tier2 chose STOP.")
                break

            # Step 4: Submit to Harness
            result, state = self.harness.execute(action.action, action.arguments, state, action.reason)

            if not result.success:
                if result.status == "BLOCKED":
                    _log.info("Harness blocked action — budget exhausted.")
                    state.status = AgentStatus.BUDGET_EXHAUSTED
                    break
                # Rejected or error — don't retry indefinitely
                _log.warning(f"Harness rejected/error: {result.reason}")
                # Count as a soft iteration but don't consume a call
                if iterations >= 3:
                    _log.warning("Too many consecutive Harness rejections — stopping.")
                    break
                continue

        # Step 5: Generate final answer (no more tool calls allowed after this point)
        _log.info("Generating final answer from evidence.")
        return self.provider.generate_final_answer(state)
