"""
Laya — Tier 1 agent.

Responsibilities:
1. Understand the question.
2. Decide HANDLE or ESCALATE.
3. If HANDLE: execute lightweight evidence acquisition through the Harness.
4. Detect emerging complexity mid-execution and escalate without losing state.
5. Never directly call document tools — always goes through Harness.
"""
import logging
from typing import List, Dict, Optional, Tuple

from app.schemas.state import AgentState, AgentStatus, Tier
from app.schemas.responses import LayaDecision, LayaDecisionType
from app.schemas.actions import AgentAction
from app.harness.harness import Harness, HarnessResult
from app.providers.base import LLMProvider

_log = logging.getLogger("rap.laya")

MAX_TIER1_CALLS = 3  # Laya uses at most half the budget before re-evaluating


class LayaAgent:
    def __init__(self, harness: Harness, provider: LLMProvider):
        self.harness = harness
        self.provider = provider

    def run(self, state: AgentState, doc_metadata: List[Dict]) -> Tuple[AgentState, Optional[LayaDecision], bool]:
        """
        Returns: (updated_state, laya_decision, should_escalate)
        """
        # Step 1: Get Laya's initial decision
        decision = self.provider.generate_laya_decision(state.question, doc_metadata)
        _log.info(f"Laya decision: {decision.decision.value} confidence={decision.confidence:.2f} reason={decision.reason!r}")

        if decision.decision == LayaDecisionType.ESCALATE:
            state = state.model_copy(deep=True)
            state.escalation_reason = decision.reason
            return state, decision, True

        # Step 2: Laya handles — lightweight execution loop
        state = state.model_copy(deep=True)
        state.current_tier = Tier.TIER_1

        for _ in range(MAX_TIER1_CALLS):
            if state.calls_remaining == 0:
                state.status = AgentStatus.BUDGET_EXHAUSTED
                return state, decision, False

            action = self.provider.generate_laya_action(state, doc_metadata)

            if action.action == "STOP":
                _log.info("Laya chose STOP.")
                break

            result, state = self.harness.execute(action.action, action.arguments, state, action.reason)

            if not result.success and result.status == "BLOCKED":
                state.status = AgentStatus.BUDGET_EXHAUSTED
                break

            # Step 3: Detect emerging complexity after each retrieval
            escalate, escalation_reason = self._detect_complexity(state)
            if escalate:
                _log.info(f"Laya escalating mid-execution: {escalation_reason}")
                state = state.model_copy(deep=True)
                state.escalation_reason = escalation_reason
                state.current_tier = Tier.TIER_2
                return state, decision, True

        return state, decision, False

    def _detect_complexity(self, state: AgentState) -> Tuple[bool, str]:
        """Check if retrieved evidence reveals complexity requiring Tier 2."""
        # Multiple keyword results on very different pages suggest policy spread
        kw_evidence = [e for e in state.evidence if e.tool == "search_keyword"]
        for ev in kw_evidence:
            if isinstance(ev.result, list) and len(ev.result) >= 3:
                return True, f"Keyword found on {len(ev.result)} pages — may require cross-page reasoning."

        # Multiple get_page results with potential conflict
        page_texts = state.get_all_page_texts()
        if len(page_texts) >= 2:
            # Heuristic: if we have contradictory signals, escalate
            combined = " ".join(page_texts).lower()
            temporal_signals = ["effective", "supersede", "replaces", "amended", "revised", "previous policy", "new policy"]
            if any(s in combined for s in temporal_signals):
                return True, "Temporal/supersession language detected — requires Tier 2 reasoning."

        return False, ""
