"""Tests for the six-call budget enforcement."""
import pytest
from app.harness.budget import BudgetEnforcer, MAX_TOOL_CALLS
from app.schemas.state import AgentState


def make_state(calls_used=0):
    return AgentState(
        question_id="Q_test", question="test", document_id="doc_001",
        calls_used=calls_used, calls_remaining=MAX_TOOL_CALLS - calls_used,
    )


def test_can_execute_at_zero():
    enforcer = BudgetEnforcer()
    assert enforcer.can_execute(make_state(0)) is True


def test_can_execute_at_five():
    enforcer = BudgetEnforcer()
    assert enforcer.can_execute(make_state(5)) is True


def test_cannot_execute_at_six():
    enforcer = BudgetEnforcer()
    assert enforcer.can_execute(make_state(6)) is False


def test_consume_increments_calls():
    enforcer = BudgetEnforcer()
    state = make_state(0)
    new_state = enforcer.consume(state)
    assert new_state.calls_used == 1
    assert new_state.calls_remaining == 5


def test_consume_does_not_mutate_original():
    enforcer = BudgetEnforcer()
    state = make_state(0)
    new_state = enforcer.consume(state)
    assert state.calls_used == 0  # original unchanged


def test_max_tool_calls_is_six():
    assert MAX_TOOL_CALLS == 6


def test_seventh_call_blocked(harness, fresh_state):
    """TEST 5: The seventh document tool call must be blocked."""
    state = fresh_state
    # 5 distinct pages + 1 keyword search = 6 real calls (no cache hits)
    actions = [
        ("get_page", {"doc_id": "doc_001", "page_number": 1}),
        ("get_page", {"doc_id": "doc_001", "page_number": 2}),
        ("get_page", {"doc_id": "doc_001", "page_number": 3}),
        ("get_page", {"doc_id": "doc_001", "page_number": 4}),
        ("get_page", {"doc_id": "doc_001", "page_number": 5}),
        ("search_keyword", {"doc_id": "doc_001", "keyword": "leave"}),
    ]
    for action, args in actions:
        result, state = harness.execute(action, args, state, "test")
        assert result.status in ("SUCCESS", "CACHE_HIT")

    assert state.calls_used == 6

    # Call #7 must be blocked
    result, state = harness.execute("search_keyword", {"doc_id": "doc_001", "keyword": "policy"}, state, "test")
    assert result.status == "BLOCKED"
    assert result.reason == "CALL_BUDGET_EXCEEDED"


def test_six_calls_exactly_allowed(harness, mock_provider, simple_provider):
    """TEST 4: Exactly 6 calls are allowed."""
    from app.tools.registry import ToolRegistry
    from app.harness.harness import Harness
    registry = ToolRegistry(simple_provider)
    h = Harness(registry)
    state = AgentState(
        question_id="Q_budget", question="test", document_id="doc_001",
        calls_used=0, calls_remaining=6,
    )
    pages = [1, 2, 3, 4, 5]
    for pg in pages:
        result, state = h.execute("get_page", {"doc_id": "doc_001", "page_number": pg}, state, "test")
        assert result.status in ("SUCCESS", "CACHE_HIT")
    # 6th call
    result, state = h.execute("search_keyword", {"doc_id": "doc_001", "keyword": "policy"}, state, "test")
    assert result.status in ("SUCCESS", "CACHE_HIT")
    assert state.calls_used == 6
