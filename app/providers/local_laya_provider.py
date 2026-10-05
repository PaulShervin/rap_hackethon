"""
Local Laya Provider — Tier 1 decision and lightweight action agent.

Uses Ollama with a configurable lightweight model.
Environment variables:
    LAYA_ENDPOINT  — Ollama base URL for Laya (default: OLLAMA_BASE_URL or http://localhost:11434)
    LAYA_MODEL     — Model name for Laya (default: gemma3)

If Laya's Ollama call fails, a conservative heuristic fallback is used
that classifies all but the simplest questions as ESCALATE.
"""
import logging
import os
from typing import Dict, List

from app.providers.base import LLMProvider
from app.providers.ollama_client import call_ollama, extract_json, check_ollama_available
from app.schemas.actions import AgentAction, ALLOWED_TOOLS
from app.schemas.responses import (
    FinalAnswer, LayaDecision, LayaDecisionType,
    VerificationResult, VerificationStatus,
)
from app.schemas.state import AgentState
from app.harness.security import SecurityPolicy

_log = logging.getLogger("rap.laya_provider")
_security = SecurityPolicy()

SIMPLE_QUESTION_SIGNALS = [
    "what is", "how many", "when is", "who is", "define", "what does",
    "is there", "does the", "list the",
]

COMPLEX_QUESTION_SIGNALS = [
    "compare", "difference between", "which applies", "supersede", "effective",
    "after the revision", "old vs new", "current vs previous", "policy change",
    "conflict", "contradict", "amended", "revised", "replace",
]


def _get_laya_config() -> tuple[str, str]:
    base_url = os.environ.get(
        "LAYA_ENDPOINT",
        os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    ).strip()
    model = os.environ.get(
        "LAYA_MODEL",
        os.environ.get("OLLAMA_MODEL", "gemma3"),
    ).strip()
    return base_url, model


def _heuristic_decision(question: str) -> LayaDecision:
    """Conservative fallback when Ollama is unavailable."""
    q = question.lower()
    for sig in COMPLEX_QUESTION_SIGNALS:
        if sig in q:
            return LayaDecision(
                decision=LayaDecisionType.ESCALATE,
                confidence=0.75,
                reason=f"Heuristic: complexity signal '{sig}' detected.",
                complexity_signals=[sig],
            )
    word_count = len(q.split())
    if word_count <= 10 and any(s in q for s in SIMPLE_QUESTION_SIGNALS):
        return LayaDecision(
            decision=LayaDecisionType.HANDLE,
            confidence=0.60,
            reason="Heuristic: short factual question, no complexity signals.",
        )
    # Default conservative: escalate
    return LayaDecision(
        decision=LayaDecisionType.ESCALATE,
        confidence=0.65,
        reason="Heuristic fallback: escalating to Tier 2 for safety.",
    )


class LocalLayaProvider(LLMProvider):
    """Laya Tier 1 provider backed by a local Ollama model."""

    def __init__(self):
        self._base_url, self._model = _get_laya_config()
        ok, msg = check_ollama_available(self._base_url, self._model)
        if not ok:
            _log.warning(f"Laya provider startup check: {msg}. Will use heuristic fallback.")
        else:
            _log.info(f"Laya provider ready: {self._model} @ {self._base_url}")
        self._available = ok

    # -----------------------------------------------------------------------
    # Tier 1 — Laya decision
    # -----------------------------------------------------------------------

    def generate_laya_decision(self, question: str, doc_metadata: List[Dict]) -> LayaDecision:
        if not self._available:
            _log.warning("Laya Ollama unavailable — using heuristic decision.")
            return _heuristic_decision(question)

        docs_summary = str(doc_metadata)
        prompt = f"""{_security.build_system_boundary_prompt()}

You are Laya, a first-level decision agent for a document-answering system.

Your task: analyze the question and decide whether to HANDLE it yourself (simple, direct factual lookup requiring at most 2-3 tool calls) or ESCALATE to a more powerful reasoning agent (complex multi-step reasoning required).

Documents available: {docs_summary}

Question: {question}

ESCALATE when you detect any of these complexity signals:
- Requires comparing conflicting statements on different pages
- Involves temporal reasoning (effective dates, revisions, supersession)
- Requires understanding which of two policies currently applies
- Multiple conditions or exceptions
- Policy interpretation or nuance
- Estimated evidence > 3 pages

HANDLE only for simple direct factual lookups like "What is the X?" where the answer is likely a single fact on one page.

Respond with ONLY this JSON (no explanation outside the JSON):
{{
  "decision": "HANDLE" or "ESCALATE",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "complexity_signals": ["<signal if any>"]
}}
"""
        try:
            raw = call_ollama(self._base_url, self._model, prompt)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in Laya decision response: {raw[:200]}")
            decision_val = str(data.get("decision", "ESCALATE")).upper()
            if decision_val not in ("HANDLE", "ESCALATE"):
                decision_val = "ESCALATE"
            return LayaDecision(
                decision=LayaDecisionType(decision_val),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
                complexity_signals=data.get("complexity_signals", []),
            )
        except RuntimeError as exc:
            _log.warning(f"Laya Ollama error: {exc}. Using heuristic fallback.")
            return _heuristic_decision(question)
        except Exception as exc:
            _log.warning(f"Laya decision parsing error: {exc}. Using heuristic fallback.")
            return _heuristic_decision(question)

    # -----------------------------------------------------------------------
    # Tier 1 — Laya action
    # -----------------------------------------------------------------------

    def generate_laya_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        if not self._available:
            _log.warning("Laya Ollama unavailable — defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason="Laya unavailable, stopping.")

        evidence_ctx = _build_light_evidence(state)
        prompt = f"""{_security.build_system_boundary_prompt()}

You are Laya, handling a simple document question. Choose the next tool call.

ALLOWED TOOLS ONLY:
- list_documents()  → {{}}
- list_headings(doc_id) → {{"doc_id": "..."}}
- search_keyword(doc_id, keyword) → {{"doc_id": "...", "keyword": "..."}} returns page numbers only
- get_page(doc_id, page_number) → {{"doc_id": "...", "page_number": <int>}} returns one page

Question: {state.question}
Calls used: {state.calls_used}/6
Calls remaining: {state.calls_remaining}
Keywords already searched: {state.searched_keywords}
Pages already retrieved: {state.retrieved_pages}
Evidence so far: {evidence_ctx}

If you have enough evidence or cannot make progress, output STOP.

Respond with ONLY this JSON:
{{
  "action": "<list_documents|list_headings|search_keyword|get_page|STOP>",
  "arguments": {{}},
  "reason": "<why>"
}}
"""
        try:
            raw = call_ollama(self._base_url, self._model, prompt)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in Laya action response: {raw[:200]}")
            action_val = str(data.get("action", "STOP"))
            if action_val not in ALLOWED_TOOLS:
                raise ValueError(f"Laya returned forbidden action: {action_val!r}")
            return AgentAction(
                action=action_val,
                arguments=data.get("arguments", {}),
                reason=str(data.get("reason", "")),
            )
        except (RuntimeError, ValueError) as exc:
            _log.warning(f"Laya action error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Action error: {exc}")
        except Exception as exc:
            _log.warning(f"Laya action unexpected error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Unexpected error: {exc}")

    # -----------------------------------------------------------------------
    # These are Tier 2 responsibilities — delegate to raise if called on Laya
    # -----------------------------------------------------------------------

    def generate_tier2_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        raise NotImplementedError("Tier 2 actions must use OllamaGemmaProvider.")

    def generate_final_answer(self, state: AgentState) -> FinalAnswer:
        raise NotImplementedError("Final answer must use OllamaGemmaProvider.")

    def verify_answerability(self, state: AgentState) -> VerificationResult:
        raise NotImplementedError("Verification must use OllamaGemmaProvider.")


def _build_light_evidence(state: AgentState) -> str:
    summaries = []
    for ev in state.evidence:
        if ev.tool == "search_keyword":
            summaries.append(f"search_keyword({ev.arguments.get('keyword')!r}) → pages {ev.result}")
        elif ev.tool == "get_page":
            text = ev.result.get("text", "")[:120] if isinstance(ev.result, dict) else ""
            summaries.append(f"page {ev.arguments.get('page_number')}: {text!r}...")
        elif ev.tool == "list_documents":
            summaries.append(f"list_documents → {ev.result}")
        elif ev.tool == "list_headings":
            summaries.append(f"list_headings → {ev.result}")
    return "; ".join(summaries) if summaries else "(none)"
