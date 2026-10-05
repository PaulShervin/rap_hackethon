"""Tests for the Deterministic Harness."""
import pytest
from app.harness.harness import Harness
from app.tools.registry import ToolRegistry
from app.schemas.state import AgentState, AgentStatus


def test_harness_executes_list_documents(harness, fresh_state):
    result, state = harness.execute("list_documents", {}, fresh_state, "test")
    assert result.success is True
    assert isinstance(result.result, list)
    assert state.calls_used == 1


def test_harness_executes_search_keyword(harness, fresh_state):
    result, state = harness.execute("search_keyword", {"doc_id": "doc_001", "keyword": "leave"}, fresh_state, "test")
    assert result.success is True
    assert isinstance(result.result, list)
    assert state.calls_used == 1
    assert "leave" in state.searched_keywords


def test_harness_executes_get_page(harness, fresh_state):
    result, state = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, fresh_state, "test")
    assert result.success is True
    assert isinstance(result.result, dict)
    assert "text" in result.result
    assert state.calls_used == 1
    assert 1 in state.retrieved_pages


def test_harness_rejects_forbidden_tool(harness, fresh_state):
    """TEST 12: Unauthorized tool must be rejected."""
    result, state = harness.execute("download_pdf", {}, fresh_state, "test")
    assert result.success is False
    assert result.status == "REJECTED"
    assert state.calls_used == 0  # budget not consumed


def test_harness_rejects_semantic_search(harness, fresh_state):
    """TEST 14: semantic_search must be rejected."""
    result, state = harness.execute("semantic_search", {"query": "policy"}, fresh_state, "test")
    assert result.success is False
    assert result.status == "REJECTED"


def test_harness_rejects_invalid_page(harness, fresh_state):
    """TEST 13: Invalid page number must be rejected."""
    result, state = harness.execute("get_page", {"doc_id": "doc_001", "page_number": -1}, fresh_state, "test")
    assert result.success is False
    assert result.status == "REJECTED"
    assert state.calls_used == 0


def test_harness_rejects_empty_keyword(harness, fresh_state):
    result, state = harness.execute("search_keyword", {"doc_id": "doc_001", "keyword": ""}, fresh_state, "test")
    assert result.success is False
    assert result.status == "REJECTED"


def test_harness_rejects_malformed_action(harness, fresh_state):
    """TEST 11: Malformed action must be rejected by Harness."""
    result, state = harness.execute("", {}, fresh_state, "test")
    assert result.success is False
    assert result.status == "REJECTED"


def test_stop_action_not_a_document_call(harness, fresh_state):
    result, state = harness.execute("STOP", {}, fresh_state, "test")
    assert result.status == "STOP"
    assert state.calls_used == 0  # STOP does not consume budget


def test_harness_logs_all_calls(harness, fresh_state):
    harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, fresh_state, "test")
    trace = harness.get_trace()
    assert len(trace) == 1
    assert trace[0]["action"] == "get_page"
    assert trace[0]["status"] == "SUCCESS"


def test_harness_logs_blocked_calls(harness, fresh_state):
    state = fresh_state.model_copy(deep=True)
    state.calls_used = 6
    state.calls_remaining = 0
    result, new_state = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, state, "test")
    trace = harness.get_trace()
    blocked = [e for e in trace if e["status"] == "BLOCKED"]
    assert len(blocked) == 1
    assert blocked[0]["reason"] == "CALL_BUDGET_EXCEEDED"
