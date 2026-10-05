from app.schemas.state import AgentState

MAX_TOOL_CALLS = 6


class BudgetEnforcer:
    def __init__(self, max_calls: int = MAX_TOOL_CALLS):
        self.max_calls = max_calls

    def can_execute(self, state: AgentState) -> bool:
        return state.calls_used < self.max_calls

    def consume(self, state: AgentState) -> AgentState:
        updated = state.model_copy(deep=True)
        updated.calls_used += 1
        updated.calls_remaining = max(0, self.max_calls - updated.calls_used)
        return updated

    def budget_status(self, state: AgentState) -> str:
        return f"{state.calls_used}/{self.max_calls}"
