"""Tests for contradiction detection and answerability (TEST 8, 9)."""
import pytest
from app.verification.verifier import DeterministicVerifier
from app.schemas.state import AgentState, Evidence, Contradiction
from app.schemas.responses import VerificationStatus


def make_state_with_evidence(pages: dict, contradictions=None):
    state = AgentState(
        question_id="Q_contra", question="What is the cancellation period?",
        document_id="doc_001", calls_used=len(pages), calls_remaining=6 - len(pages),
    )
    for page_num, text in pages.items():
        ev = Evidence(
            tool="get_page",
            arguments={"doc_id": "doc_001", "page_number": page_num},
            result={"doc_id": "doc_001", "page_number": page_num, "text": text},
            call_number=list(pages.keys()).index(page_num) + 1,
        )
        state.evidence.append(ev)
        state.retrieved_pages.append(page_num)
    if contradictions:
        state.contradictions = contradictions
    return state


def test_unresolved_contradiction_gives_insufficient():
    """TEST 8: Contradictory evidence with no resolution -> INSUFFICIENT."""
    contradiction = Contradiction(
        claim_a="Cancellation period is 30 days",
        source_a="Page 4",
        claim_b="Cancellation period is 15 days",
        source_b="Page 8",
        resolved=False,
    )
    state = make_state_with_evidence(
        {4: "Cancellation: 30 days.", 8: "Cancellation: 15 days."},
        contradictions=[contradiction],
    )
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    assert result.status == VerificationStatus.INSUFFICIENT
    assert len(result.unresolved_conflicts) >= 1


def test_resolved_contradiction_passes():
    """TEST 9: When supersession is clear, contradiction is resolved."""
    contradiction = Contradiction(
        claim_a="30 days",
        source_a="Page 4",
        claim_b="15 days effective Jan 2026",
        source_b="Page 8",
        resolved=True,
        resolution="Page 9 explicitly states this policy supersedes page 4.",
    )
    state = make_state_with_evidence(
        {4: "Cancellation: 30 days.", 8: "15 days effective Jan 2026.", 9: "supersedes previous"},
        contradictions=[contradiction],
    )
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    # Resolved contradictions pass deterministic check
    assert result.status == VerificationStatus.ANSWERABLE


def test_no_pages_retrieved_gives_insufficient():
    state = AgentState(
        question_id="Q_nopages", question="test", document_id="doc_001",
        calls_used=1, calls_remaining=5,
    )
    # Only keyword search, no get_page
    ev = Evidence(
        tool="search_keyword",
        arguments={"doc_id": "doc_001", "keyword": "cancellation"},
        result=[4, 8],
        call_number=1,
    )
    state.evidence.append(ev)
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    assert result.status == VerificationStatus.INSUFFICIENT


def test_budget_exhausted_no_evidence_is_insufficient():
    state = AgentState(
        question_id="Q_empty", question="test", document_id="doc_001",
        calls_used=6, calls_remaining=0,
    )
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    assert result.status == VerificationStatus.INSUFFICIENT
