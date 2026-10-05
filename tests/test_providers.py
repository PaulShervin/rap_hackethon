"""
Tests for local providers (Laya + OllamaGemma).
Mocks the Ollama HTTP client so tests run without Ollama installed.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.providers.local_laya_provider import LocalLayaProvider, _heuristic_decision
from app.providers.ollama_provider import OllamaGemmaProvider
from app.schemas.responses import LayaDecisionType, VerificationStatus
from app.schemas.state import AgentState
from app.schemas.actions import AgentAction


# ---------------------------------------------------------------------------
# Heuristic decision tests (no Ollama needed)
# ---------------------------------------------------------------------------

def test_heuristic_simple_question():
    decision = _heuristic_decision("What is the maximum leave period?")
    assert decision.decision in (LayaDecisionType.HANDLE, LayaDecisionType.ESCALATE)
    assert 0.0 <= decision.confidence <= 1.0


def test_heuristic_complex_question():
    decision = _heuristic_decision("Compare the old and revised cancellation policies and which supersedes the other?")
    assert decision.decision == LayaDecisionType.ESCALATE


def test_heuristic_temporal_escalate():
    decision = _heuristic_decision("Which policy applies after the effective date revision?")
    assert decision.decision == LayaDecisionType.ESCALATE


def test_heuristic_returns_reason():
    decision = _heuristic_decision("What is the leave period?")
    assert decision.reason != ""


# ---------------------------------------------------------------------------
# LocalLayaProvider with mocked Ollama
# ---------------------------------------------------------------------------

LAYA_DECISION_JSON = '{"decision": "HANDLE", "confidence": 0.85, "reason": "Simple factual lookup", "complexity_signals": []}'
LAYA_ESCALATE_JSON = '{"decision": "ESCALATE", "confidence": 0.90, "reason": "Requires temporal reasoning", "complexity_signals": ["effective dates"]}'
LAYA_ACTION_JSON = '{"action": "search_keyword", "arguments": {"doc_id": "doc_001", "keyword": "leave"}, "reason": "Find leave policy pages"}'
LAYA_STOP_JSON = '{"action": "STOP", "arguments": {}, "reason": "Sufficient evidence"}'


@pytest.fixture
def laya_provider():
    with patch("app.providers.local_laya_provider.check_ollama_available", return_value=(True, "ok")):
        provider = LocalLayaProvider()
    provider._available = True
    return provider


def test_laya_decision_handle(laya_provider):
    with patch("app.providers.local_laya_provider.call_ollama", return_value=LAYA_DECISION_JSON):
        decision = laya_provider.generate_laya_decision("What is the leave period?", [])
    assert decision.decision == LayaDecisionType.HANDLE
    assert decision.confidence == 0.85


def test_laya_decision_escalate(laya_provider):
    with patch("app.providers.local_laya_provider.call_ollama", return_value=LAYA_ESCALATE_JSON):
        decision = laya_provider.generate_laya_decision("Which policy applies after revision?", [])
    assert decision.decision == LayaDecisionType.ESCALATE
    assert "effective dates" in decision.complexity_signals


def test_laya_decision_falls_back_on_bad_json(laya_provider):
    with patch("app.providers.local_laya_provider.call_ollama", return_value="not json at all"):
        decision = laya_provider.generate_laya_decision("test?", [])
    # Should fall back to heuristic — not raise
    assert decision.decision in (LayaDecisionType.HANDLE, LayaDecisionType.ESCALATE)


def test_laya_decision_falls_back_on_connection_error(laya_provider):
    with patch("app.providers.local_laya_provider.call_ollama", side_effect=RuntimeError("connection refused")):
        decision = laya_provider.generate_laya_decision("test?", [])
    assert decision.decision in (LayaDecisionType.HANDLE, LayaDecisionType.ESCALATE)


def test_laya_action_search_keyword(laya_provider):
    state = AgentState(question_id="Q1", question="test", document_id="doc_001",
                       calls_used=0, calls_remaining=6)
    with patch("app.providers.local_laya_provider.call_ollama", return_value=LAYA_ACTION_JSON):
        action = laya_provider.generate_laya_action(state, [{"doc_id": "doc_001"}])
    assert action.action == "search_keyword"
    assert action.arguments["keyword"] == "leave"


def test_laya_action_stop(laya_provider):
    state = AgentState(question_id="Q1", question="test", document_id="doc_001",
                       calls_used=5, calls_remaining=1)
    with patch("app.providers.local_laya_provider.call_ollama", return_value=LAYA_STOP_JSON):
        action = laya_provider.generate_laya_action(state, [])
    assert action.action == "STOP"


def test_laya_action_rejects_forbidden_tool(laya_provider):
    state = AgentState(question_id="Q1", question="test", document_id="doc_001",
                       calls_used=0, calls_remaining=6)
    bad_json = '{"action": "download_pdf", "arguments": {}, "reason": "test"}'
    with patch("app.providers.local_laya_provider.call_ollama", return_value=bad_json):
        action = laya_provider.generate_laya_action(state, [])
    # Forbidden action → fallback to STOP
    assert action.action == "STOP"


def test_laya_unavailable_returns_heuristic():
    with patch("app.providers.local_laya_provider.check_ollama_available", return_value=(False, "offline")):
        provider = LocalLayaProvider()
    decision = provider.generate_laya_decision("What is the leave period?", [])
    assert decision.decision in (LayaDecisionType.HANDLE, LayaDecisionType.ESCALATE)


def test_laya_tier2_methods_raise():
    with patch("app.providers.local_laya_provider.check_ollama_available", return_value=(True, "ok")):
        provider = LocalLayaProvider()
    state = AgentState(question_id="Q1", question="test", document_id="doc_001",
                       calls_used=0, calls_remaining=6)
    with pytest.raises(NotImplementedError):
        provider.generate_tier2_action(state, [])
    with pytest.raises(NotImplementedError):
        provider.generate_final_answer(state)


# ---------------------------------------------------------------------------
# OllamaGemmaProvider with mocked Ollama
# ---------------------------------------------------------------------------

TIER2_ACTION_JSON = '{"action": "get_page", "arguments": {"doc_id": "doc_001", "page_number": 4}, "reason": "Check page 4 for policy"}'
VERIFICATION_ANSWERABLE_JSON = '{"status": "ANSWERABLE", "supported_claims": ["30 days"], "unresolved_conflicts": [], "reason": "Evidence supports answer"}'
VERIFICATION_INSUFFICIENT_JSON = '{"status": "INSUFFICIENT", "supported_claims": [], "unresolved_conflicts": ["30 vs 15 days conflict"], "reason": "Contradictory evidence"}'
FINAL_ANSWER_JSON = '{"answer": "The leave period is 30 days. Sources: Page 2.", "sources": ["Page 2"]}'


@pytest.fixture
def tier2_provider():
    with patch("app.providers.ollama_provider.check_ollama_available", return_value=(True, "ok")):
        provider = OllamaGemmaProvider()
    provider._available = True
    return provider


def _state_with_page(page_num=2, text="Leave period is 30 days."):
    from app.schemas.state import Evidence
    state = AgentState(question_id="Q2", question="What is the leave period?",
                       document_id="doc_001", calls_used=1, calls_remaining=5)
    ev = Evidence(tool="get_page", arguments={"doc_id": "doc_001", "page_number": page_num},
                  result={"doc_id": "doc_001", "page_number": page_num, "text": text}, call_number=1)
    state.evidence.append(ev)
    state.retrieved_pages.append(page_num)
    return state


def test_tier2_action_get_page(tier2_provider):
    state = AgentState(question_id="Q2", question="test", document_id="doc_001",
                       calls_used=1, calls_remaining=5)
    with patch("app.providers.ollama_provider.call_ollama", return_value=TIER2_ACTION_JSON):
        action = tier2_provider.generate_tier2_action(state, [])
    assert action.action == "get_page"
    assert action.arguments["page_number"] == 4


def test_tier2_action_rejects_forbidden_tool(tier2_provider):
    state = AgentState(question_id="Q2", question="test", document_id="doc_001",
                       calls_used=0, calls_remaining=6)
    bad = '{"action": "semantic_search", "arguments": {"query": "policy"}, "reason": "test"}'
    with patch("app.providers.ollama_provider.call_ollama", return_value=bad):
        action = tier2_provider.generate_tier2_action(state, [])
    assert action.action == "STOP"


def test_tier2_verification_answerable(tier2_provider):
    state = _state_with_page()
    with patch("app.providers.ollama_provider.call_ollama", return_value=VERIFICATION_ANSWERABLE_JSON):
        result = tier2_provider.verify_answerability(state)
    assert result.status == VerificationStatus.ANSWERABLE


def test_tier2_verification_insufficient(tier2_provider):
    state = _state_with_page()
    with patch("app.providers.ollama_provider.call_ollama", return_value=VERIFICATION_INSUFFICIENT_JSON):
        result = tier2_provider.verify_answerability(state)
    assert result.status == VerificationStatus.INSUFFICIENT
    assert len(result.unresolved_conflicts) > 0


def test_tier2_final_answer_answerable(tier2_provider):
    state = _state_with_page()
    with patch("app.providers.ollama_provider.call_ollama", side_effect=[
        VERIFICATION_ANSWERABLE_JSON,
        FINAL_ANSWER_JSON,
    ]):
        answer = tier2_provider.generate_final_answer(state)
    assert answer.insufficient is False
    assert "30 days" in answer.answer


def test_tier2_final_answer_insufficient_when_verification_fails(tier2_provider):
    state = _state_with_page()
    with patch("app.providers.ollama_provider.call_ollama", return_value=VERIFICATION_INSUFFICIENT_JSON):
        answer = tier2_provider.generate_final_answer(state)
    assert answer.insufficient is True


def test_tier2_unavailable_returns_insufficient():
    with patch("app.providers.ollama_provider.check_ollama_available", return_value=(False, "offline")):
        provider = OllamaGemmaProvider()
    state = _state_with_page()
    answer = provider.generate_final_answer(state)
    assert answer.insufficient is True
    assert "unavailable" in answer.answer.lower()


def test_tier2_laya_methods_raise():
    with patch("app.providers.ollama_provider.check_ollama_available", return_value=(True, "ok")):
        provider = OllamaGemmaProvider()
    with pytest.raises(NotImplementedError):
        provider.generate_laya_decision("test?", [])
    with pytest.raises(NotImplementedError):
        provider.generate_laya_action(AgentState(question_id="Q1", question="test",
                                                  document_id="doc_001"), [])
