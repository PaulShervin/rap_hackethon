"""TEST 10: Information absent from PDF -> INSUFFICIENT."""
import pytest
from app.verification.verifier import DeterministicVerifier
from app.schemas.state import AgentState, Evidence
from app.schemas.responses import VerificationStatus


def test_question_about_absent_topic_is_insufficient():
    """Budget exhausted searching for info that doesn't exist -> INSUFFICIENT."""
    state = AgentState(
        question_id="Q_absent", question="What is the bonus structure?",
        document_id="doc_001", calls_used=6, calls_remaining=0,
    )
    # Only search results returned, keyword never found
    ev = Evidence(
        tool="search_keyword",
        arguments={"doc_id": "doc_001", "keyword": "bonus"},
        result=[],  # not found on any page
        call_number=1,
    )
    state.evidence.append(ev)
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    assert result.status == VerificationStatus.INSUFFICIENT


def test_with_pages_retrieved_passes_deterministic_check():
    state = AgentState(
        question_id="Q_ok", question="test", document_id="doc_001",
        calls_used=2, calls_remaining=4,
    )
    ev = Evidence(
        tool="get_page",
        arguments={"doc_id": "doc_001", "page_number": 3},
        result={"doc_id": "doc_001", "page_number": 3, "text": "The leave period is 30 days."},
        call_number=1,
    )
    state.evidence.append(ev)
    verifier = DeterministicVerifier()
    result = verifier.check(state)
    assert result.status == VerificationStatus.ANSWERABLE
