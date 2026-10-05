"""TEST 6: Duplicate retrieval prevention."""
import pytest
from app.harness.cache import EvidenceCache
from app.schemas.state import AgentState, Evidence


def make_state_with_page(doc_id="doc_001", page_number=8, text="some text"):
    ev = Evidence(
        evidence_id="E1",
        tool="get_page",
        arguments={"doc_id": doc_id, "page_number": page_number},
        result={"doc_id": doc_id, "page_number": page_number, "text": text},
        call_number=1,
    )
    state = AgentState(
        question_id="Q_test", question="test", document_id=doc_id,
        calls_used=1, calls_remaining=5,
    )
    state.evidence.append(ev)
    return state


def test_cache_hit_for_get_page():
    cache = EvidenceCache()
    state = make_state_with_page(page_number=8)
    hit = cache.lookup(state, "get_page", {"doc_id": "doc_001", "page_number": 8})
    assert hit is not None
    assert hit.result["page_number"] == 8


def test_cache_miss_for_different_page():
    cache = EvidenceCache()
    state = make_state_with_page(page_number=8)
    hit = cache.lookup(state, "get_page", {"doc_id": "doc_001", "page_number": 9})
    assert hit is None


def test_cache_hit_does_not_consume_budget(harness, fresh_state):
    # First call — real execution, costs 1
    result, state = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, fresh_state, "first")
    assert state.calls_used == 1

    # Second call — same page, should be cache hit
    result2, state2 = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, state, "second")
    assert result2.cached is True
    assert result2.status == "CACHE_HIT"
    assert state2.calls_used == 1  # no additional budget consumed


def test_cache_miss_consumes_budget(harness, fresh_state):
    result, state = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 1}, fresh_state, "first")
    assert state.calls_used == 1
    result2, state2 = harness.execute("get_page", {"doc_id": "doc_001", "page_number": 2}, state, "second")
    assert result2.cached is False
    assert state2.calls_used == 2


def test_cache_does_not_provide_unretreived_data():
    cache = EvidenceCache()
    state = AgentState(
        question_id="Q_test", question="test", document_id="doc_001",
        calls_used=0, calls_remaining=6,
    )
    hit = cache.lookup(state, "get_page", {"doc_id": "doc_001", "page_number": 99})
    assert hit is None
