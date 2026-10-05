"""
The Deterministic Harness.

"The agent decides what should happen.
 The Harness decides what is allowed to happen."

This is the ONLY component that executes document tools.
No LLM lives inside this class.
"""
from typing import Any, Dict, Optional, Tuple

from app.schemas.actions import DOCUMENT_TOOLS
from app.schemas.state import AgentState, AgentStatus, Evidence
from app.harness.budget import BudgetEnforcer, MAX_TOOL_CALLS
from app.harness.cache import EvidenceCache
from app.harness.validator import ActionValidator
from app.harness.security import SecurityPolicy
from app.harness.logger import HarnessLogger
from app.tools.registry import ToolRegistry


class HarnessResult:
    def __init__(self, success: bool, result: Any, status: str, reason: str,
                 cached: bool = False, state: Optional[AgentState] = None):
        self.success = success
        self.result = result
        self.status = status
        self.reason = reason
        self.cached = cached
        self.state = state


class Harness:
    def __init__(self, tool_registry: ToolRegistry, max_calls: int = MAX_TOOL_CALLS):
        self.registry = tool_registry
        self.budget = BudgetEnforcer(max_calls)
        self.cache = EvidenceCache()
        self.validator = ActionValidator()
        self.security = SecurityPolicy()
        self.logger = HarnessLogger()

    def execute(self, action: str, arguments: Dict[str, Any], state: AgentState, reason: str = "") -> Tuple[HarnessResult, AgentState]:
        question_id = state.question_id
        tier = state.current_tier.value

        # Step 1: Validate action structure and tool allowlist
        valid, validation_msg = self.validator.validate(action, arguments)
        if not valid:
            self.logger.log_rejected(question_id, tier, action, arguments, validation_msg)
            return HarnessResult(False, None, "REJECTED", validation_msg), state

        # Step 2: STOP is not a document tool call — return immediately
        if action == "STOP":
            self.logger.log_success(question_id, tier, state.calls_used, "STOP", {}, "Agent requested stop.", reason, state.calls_remaining)
            return HarnessResult(True, None, "STOP", reason), state

        # Step 3: Check six-call budget BEFORE executing any document tool
        if not self.budget.can_execute(state):
            block_reason = "CALL_BUDGET_EXCEEDED"
            self.logger.log_blocked(question_id, tier, state.calls_used + 1, action, arguments, block_reason, 0)
            new_state = state.model_copy(deep=True)
            new_state.status = AgentStatus.BUDGET_EXHAUSTED
            return HarnessResult(False, None, "BLOCKED", block_reason), new_state

        # Step 4: Check evidence cache (duplicate retrieval prevention)
        cached_evidence = self.cache.lookup(state, action, arguments)
        if cached_evidence is not None:
            self.logger.log_success(
                question_id, tier, state.calls_used, action, arguments,
                f"Cache hit (no budget consumed): evidence_id={cached_evidence.evidence_id}",
                reason, state.calls_remaining, cached=True,
            )
            # Return cached result WITHOUT consuming budget
            return HarnessResult(True, cached_evidence.result, "CACHE_HIT", "Returned from question-scoped cache.", cached=True, state=state), state

        # Step 5: Execute tool
        try:
            result = self.registry.execute(action, arguments)
        except Exception as exc:
            err_reason = f"Tool execution error: {exc}"
            self.logger.log_rejected(question_id, tier, action, arguments, err_reason)
            return HarnessResult(False, None, "ERROR", err_reason), state

        # Step 6: Consume budget and update state
        new_state = self.budget.consume(state)
        call_num = new_state.calls_used

        # Step 7: Build result summary for logging (sanitized)
        result_summary = self._summarize_result(action, result)

        self.logger.log_success(
            question_id, tier, call_num, action, arguments,
            result_summary, reason, new_state.calls_remaining,
        )

        # Step 8: Store evidence in state
        evidence = Evidence(
            tool=action,
            arguments=arguments,
            result=result,
            call_number=call_num,
        )
        new_state = new_state.model_copy(deep=True)
        new_state.evidence.append(evidence)
        new_state.last_action = {"action": action, "arguments": arguments}
        new_state.last_tool_result = result

        # Step 9: Update convenience tracking fields
        if action == "search_keyword":
            kw = arguments.get("keyword", "")
            if kw and kw not in new_state.searched_keywords:
                new_state.searched_keywords.append(kw)
        elif action == "get_page":
            pg = arguments.get("page_number")
            if pg and pg not in new_state.retrieved_pages:
                new_state.retrieved_pages.append(pg)

        return HarnessResult(True, result, "SUCCESS", reason), new_state

    def _summarize_result(self, action: str, result: Any) -> str:
        if action == "list_documents":
            if isinstance(result, list):
                return f"Listed {len(result)} document(s)."
            return "Listed documents."
        if action == "list_headings":
            if isinstance(result, list):
                return f"Retrieved {len(result)} heading(s)."
            return "Retrieved headings."
        if action == "search_keyword":
            if isinstance(result, list):
                return f"Keyword found on pages: {result}"
            return "Keyword search complete."
        if action == "get_page":
            if isinstance(result, dict):
                text = result.get("text", "")
                return f"Page retrieved. Text length: {len(text)} chars."
            return "Page retrieved."
        return "Tool executed."

    def get_trace(self):
        return self.logger.get_trace()

    def reset_trace(self):
        self.logger.clear()
