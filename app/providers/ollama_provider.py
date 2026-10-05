"""
Ollama + Gemma 3 Provider — Tier 2 full reasoning agent.

Environment variables:
    OLLAMA_BASE_URL  — Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL     — Model name (default: gemma3)

Gemma 3 is responsible for multi-step reasoning, contradiction resolution,
temporal reasoning, supersession detection, and final grounded answers.

Gemma NEVER directly executes document tools. It outputs structured JSON actions
which the Harness validates and executes.
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

_log = logging.getLogger("rap.ollama_provider")
_security = SecurityPolicy()


def _get_ollama_config() -> tuple[str, str]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    model = os.environ.get("OLLAMA_MODEL", "gemma3").strip()
    return base_url, model


def _build_evidence_context(state: AgentState) -> str:
    parts = []
    for ev in state.evidence:
        if ev.tool == "get_page" and isinstance(ev.result, dict):
            fenced = _security.build_evidence_fence(ev.result.get("text", ""))
            parts.append(f"[Page {ev.result.get('page_number')}]\n{fenced}")
        elif ev.tool == "search_keyword":
            parts.append(
                f"[search_keyword({ev.arguments.get('keyword')!r})] → pages: {ev.result}"
            )
        elif ev.tool == "list_documents":
            parts.append(f"[list_documents] → {ev.result}")
        elif ev.tool == "list_headings":
            parts.append(
                f"[list_headings({ev.arguments.get('doc_id')!r})] → {ev.result}"
            )
    return "\n\n".join(parts) if parts else "(no evidence retrieved yet)"


class OllamaGemmaProvider(LLMProvider):
    """Tier 2 provider: Gemma 3 via local Ollama server."""

    def __init__(self):
        self._base_url, self._model = _get_ollama_config()
        ok, msg = check_ollama_available(self._base_url, self._model)
        if not ok:
            _log.error(
                f"Tier 2 Ollama check failed: {msg}\n"
                f"Ensure Ollama is running and the model is pulled:\n"
                f"  ollama serve\n"
                f"  ollama pull {self._model}"
            )
        else:
            _log.info(f"Tier 2 provider ready: {self._model} @ {self._base_url}")
        self._available = ok
        self._model_name = self._model

    # -----------------------------------------------------------------------
    # Tier 2 action
    # -----------------------------------------------------------------------

    def generate_tier2_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        self._require_available()
        evidence_ctx = _build_evidence_context(state)
        docs_summary = str(doc_metadata)

        prompt = f"""{_security.build_system_boundary_prompt()}

You are Gemma, a document-answering reasoning agent. Your task is to answer the user's question using ONLY the four allowed document tools.

ALLOWED TOOLS (the ONLY tools you may use — no others exist):
1. list_documents()                          — get document list
2. list_headings(doc_id)                     — get heading list
3. search_keyword(doc_id, keyword)           — returns PAGE NUMBERS ONLY
4. get_page(doc_id, page_number)             — returns ONE page of text

FORBIDDEN: semantic_search, vector_search, download_pdf, read_pdf, get_all_pages, or any other tool.

Current state:
{state.summary_for_llm()}

Documents: {docs_summary}

Evidence retrieved so far:
{evidence_ctx}

IMPORTANT RULES:
- You have {state.calls_remaining} tool calls remaining. Budget wisely.
- Do NOT repeat searches/page retrievals already done — they are cached and won't consume budget, but show strategic thinking.
- If you have enough evidence to answer the question, output STOP.
- If evidence is contradictory, look for a page that clarifies which policy supersedes the other.
- If you cannot gather enough evidence within the remaining budget, output STOP and the system will return "Insufficient information."

Respond with ONLY this JSON (no explanation outside JSON):
{{
  "action": "<list_documents|list_headings|search_keyword|get_page|STOP>",
  "arguments": {{}},
  "reason": "<why this action>"
}}

For search_keyword: {{"doc_id": "...", "keyword": "..."}}
For get_page: {{"doc_id": "...", "page_number": <integer>}}
For list_headings: {{"doc_id": "..."}}
For STOP or list_documents: {{}}
"""
        try:
            raw = call_ollama(self._base_url, self._model, prompt)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in Tier 2 action response: {raw[:300]}")
            action_val = str(data.get("action", "STOP"))
            if action_val not in ALLOWED_TOOLS:
                raise ValueError(f"Gemma returned forbidden action: {action_val!r}")
            return AgentAction(
                action=action_val,
                arguments=data.get("arguments", {}),
                reason=str(data.get("reason", "")),
            )
        except (RuntimeError, ValueError) as exc:
            _log.warning(f"Tier 2 action error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Action error: {exc}")
        except Exception as exc:
            _log.warning(f"Tier 2 unexpected error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Unexpected: {exc}")

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    def verify_answerability(self, state: AgentState) -> VerificationResult:
        self._require_available()
        evidence_ctx = _build_evidence_context(state)
        contradictions_str = (
            "\n".join(
                f"- {c.claim_a!r} (source: {c.source_a}) vs {c.claim_b!r} (source: {c.source_b}), resolved: {c.resolved}"
                for c in state.contradictions
            )
            if state.contradictions
            else "(none detected)"
        )

        prompt = f"""{_security.build_system_boundary_prompt()}

You are a verification agent. Decide whether the retrieved evidence is sufficient to answer the question.

Question: {state.question}

Evidence:
{evidence_ctx}

Contradictions:
{contradictions_str}

Calls used: {state.calls_used}/6

RULES:
1. ANSWERABLE only if ALL claims in the answer are directly supported by the retrieved evidence above.
2. INSUFFICIENT if: evidence is missing, contradictory without resolution, or budget exhausted without enough data.
3. Be CONSERVATIVE. When uncertain, return INSUFFICIENT.
4. Do NOT hallucinate facts not present in the evidence.

Respond with ONLY this JSON:
{{
  "status": "ANSWERABLE" or "INSUFFICIENT",
  "supported_claims": ["<claim supported by evidence>"],
  "unresolved_conflicts": ["<conflict description if any>"],
  "reason": "<explanation>"
}}
"""
        try:
            raw = call_ollama(self._base_url, self._model, prompt, temperature=0.0)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in verification response: {raw[:300]}")
            status_val = str(data.get("status", "INSUFFICIENT")).upper()
            if status_val not in ("ANSWERABLE", "INSUFFICIENT"):
                status_val = "INSUFFICIENT"
            return VerificationResult(
                status=VerificationStatus(status_val),
                supported_claims=data.get("supported_claims", []),
                unresolved_conflicts=data.get("unresolved_conflicts", []),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            _log.warning(f"Verification error: {exc}. Defaulting to INSUFFICIENT.")
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                reason=f"Verification failed: {exc}",
            )

    # -----------------------------------------------------------------------
    # Final answer
    # -----------------------------------------------------------------------

    def generate_final_answer(self, state: AgentState) -> FinalAnswer:
        if not self._available:
            return FinalAnswer(
                answer="Insufficient information. Tier 2 model (Ollama/Gemma) is unavailable.",
                sources=[],
                insufficient=True,
            )

        evidence_ctx = _build_evidence_context(state)
        verification = self.verify_answerability(state)

        if verification.status == VerificationStatus.INSUFFICIENT:
            conflicts = "; ".join(verification.unresolved_conflicts) if verification.unresolved_conflicts else verification.reason
            return FinalAnswer(
                answer=f"Insufficient information. {conflicts}".strip(),
                sources=[],
                verification=verification,
                insufficient=True,
            )

        pages_cited = [str(p) for p in state.retrieved_pages]

        prompt = f"""{_security.build_system_boundary_prompt()}

You are generating the FINAL answer. Use ONLY the evidence below. Do NOT make additional tool calls. Do NOT hallucinate.

Question: {state.question}

Evidence:
{evidence_ctx}

Pages retrieved: {pages_cited}

RULES:
- Answer directly and concisely.
- Cite only page numbers that were actually retrieved and contain the relevant answer.
- If evidence shows a newer policy supersedes an older one, explain this clearly.
- If you cannot answer from the evidence alone, say "Insufficient information."
- End your answer with: Sources: Page X, Page Y (only pages actually relevant to the answer).
- Do NOT invent facts or page numbers not present in the evidence above.

Respond with ONLY this JSON:
{{
  "answer": "<complete answer text including Sources line>",
  "sources": ["Page X", "Page Y"]
}}
"""
        try:
            raw = call_ollama(self._base_url, self._model, prompt, temperature=0.0)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in final answer response: {raw[:300]}")
            answer_text = str(data.get("answer", "Insufficient information."))
            sources = data.get("sources", [])
            return FinalAnswer(
                answer=answer_text,
                sources=sources,
                verification=verification,
                insufficient=False,
            )
        except Exception as exc:
            _log.warning(f"Final answer generation error: {exc}")
            return FinalAnswer(
                answer="Insufficient information. Answer generation encountered an error.",
                sources=[],
                verification=verification,
                insufficient=True,
            )

    # -----------------------------------------------------------------------
    # Laya methods — not used on this provider, raise to enforce separation
    # -----------------------------------------------------------------------

    def generate_laya_decision(self, question: str, doc_metadata: List[Dict]) -> LayaDecision:
        raise NotImplementedError("Laya decisions must use LocalLayaProvider.")

    def generate_laya_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        raise NotImplementedError("Laya actions must use LocalLayaProvider.")

    def _require_available(self):
        if not self._available:
            raise RuntimeError(
                f"Ollama Tier 2 model '{self._model}' is unavailable. "
                f"Run: ollama serve && ollama pull {self._model}"
            )
