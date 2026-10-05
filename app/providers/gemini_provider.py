"""
Gemini provider — isolated Gemini API integration.
Both Laya (Tier 1) and Tier 2 use this provider; Tier 1 uses a lightweight model.
Environment variables:
    GEMINI_API_KEY   — required
    GEMINI_MODEL     — Tier 2 model (default: gemini-2.0-flash)
    LAYER1_MODEL     — Tier 1 model (default: same as GEMINI_MODEL)
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.providers.base import LLMProvider
from app.schemas.actions import AgentAction, ALLOWED_TOOLS
from app.schemas.responses import (
    FinalAnswer, LayaDecision, LayaDecisionType,
    VerificationResult, VerificationStatus,
)
from app.schemas.state import AgentState
from app.harness.security import SecurityPolicy

_log = logging.getLogger("rap.gemini")
_security = SecurityPolicy()


def _get_model(env_var: str, fallback: str) -> str:
    return os.environ.get(env_var, fallback).strip()


def _configure_genai():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)


def _call_model(model_name: str, prompt: str, temperature: float = 0.1) -> str:
    _configure_genai()
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        ),
    )
    return response.text


def _extract_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from a string that may contain prose."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block in markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to extract first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _build_evidence_context(state: AgentState) -> str:
    parts = []
    for ev in state.evidence:
        if ev.tool == "get_page" and isinstance(ev.result, dict):
            page_text = ev.result.get("text", "")
            fenced = _security.build_evidence_fence(page_text)
            parts.append(f"[Page {ev.result.get('page_number')}]\n{fenced}")
        elif ev.tool == "search_keyword" and isinstance(ev.result, list):
            parts.append(f"[search_keyword({ev.arguments.get('keyword')!r})] => pages: {ev.result}")
        elif ev.tool == "list_documents" and isinstance(ev.result, list):
            parts.append(f"[list_documents] => {ev.result}")
        elif ev.tool == "list_headings" and isinstance(ev.result, list):
            parts.append(f"[list_headings({ev.arguments.get('doc_id')!r})] => {ev.result}")
    return "\n\n".join(parts) if parts else "(no evidence retrieved yet)"


class GeminiProvider(LLMProvider):
    def __init__(self):
        self._tier2_model = _get_model("GEMINI_MODEL", "gemini-2.0-flash")
        self._tier1_model = _get_model("LAYER1_MODEL", self._tier2_model)

    # -----------------------------------------------------------------------
    # Tier 1 — Laya decision
    # -----------------------------------------------------------------------

    def generate_laya_decision(self, question: str, doc_metadata: List[Dict]) -> LayaDecision:
        docs_summary = json.dumps(doc_metadata, indent=2)
        prompt = f"""{_security.build_system_boundary_prompt()}

You are Laya, a first-level decision agent. Analyze the question and decide whether to HANDLE it yourself (simple, direct factual lookup) or ESCALATE to Tier 2 (complex reasoning required).

Documents available:
{docs_summary}

Question: {question}

Complexity signals that indicate ESCALATE:
- Requires comparing multiple conflicting statements
- Involves temporal reasoning (effective dates, revisions, supersession)
- Multiple conditions or exceptions
- Multiple documents
- Contradiction possibility
- Expected evidence count > 2 pages
- Policy interpretation or legal nuance

Output ONLY valid JSON with this exact schema:
{{
  "decision": "HANDLE" or "ESCALATE",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "complexity_signals": ["<signal1>", "<signal2>"]
}}
"""
        try:
            raw = _call_model(self._tier1_model, prompt)
            data = _extract_json(raw)
            if not data:
                raise ValueError(f"Could not parse JSON from Laya decision: {raw[:300]}")
            decision_val = data.get("decision", "ESCALATE").upper()
            if decision_val not in ("HANDLE", "ESCALATE"):
                decision_val = "ESCALATE"
            return LayaDecision(
                decision=LayaDecisionType(decision_val),
                confidence=float(data.get("confidence", 0.5)),
                reason=str(data.get("reason", "")),
                complexity_signals=data.get("complexity_signals", []),
            )
        except Exception as exc:
            _log.warning(f"Laya decision error, defaulting to ESCALATE: {exc}")
            return LayaDecision(
                decision=LayaDecisionType.ESCALATE,
                confidence=0.5,
                reason=f"Laya decision failed, escalating for safety: {exc}",
            )

    # -----------------------------------------------------------------------
    # Tier 1 — Laya action
    # -----------------------------------------------------------------------

    def generate_laya_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        return self._generate_action(state, doc_metadata, tier="TIER_1")

    # -----------------------------------------------------------------------
    # Tier 2 — action
    # -----------------------------------------------------------------------

    def generate_tier2_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        return self._generate_action(state, doc_metadata, tier="TIER_2")

    # -----------------------------------------------------------------------
    # Shared action generation
    # -----------------------------------------------------------------------

    def _generate_action(self, state: AgentState, doc_metadata: List[Dict], tier: str) -> AgentAction:
        model = self._tier1_model if tier == "TIER_1" else self._tier2_model
        evidence_ctx = _build_evidence_context(state)
        docs_summary = json.dumps(doc_metadata, indent=2)

        prompt = f"""{_security.build_system_boundary_prompt()}

You are a document-answering agent. Your goal is to answer the user's question using ONLY the four allowed document tools.

ALLOWED TOOLS (these are the ONLY tools you may use):
- list_documents() — get document metadata
- list_headings(doc_id) — get headings
- search_keyword(doc_id, keyword) — returns page numbers ONLY
- get_page(doc_id, page_number) — returns one page of text

FORBIDDEN: Any other tool, semantic search, embeddings, or direct PDF access.

Current state:
{state.summary_for_llm()}

Documents available:
{docs_summary}

Evidence retrieved so far:
{evidence_ctx}

IMPORTANT: You have {state.calls_remaining} tool calls remaining. Budget wisely.
If you have enough evidence to answer the question, output action "STOP".
If evidence is clearly insufficient and budget is low, output "STOP" so the system returns "Insufficient information."

Output ONLY valid JSON:
{{
  "action": "<one of: list_documents, list_headings, search_keyword, get_page, STOP>",
  "arguments": {{}},
  "reason": "<why this action>"
}}

For search_keyword: {{"doc_id": "...", "keyword": "..."}}
For get_page: {{"doc_id": "...", "page_number": <integer>}}
For list_headings: {{"doc_id": "..."}}
For STOP: {{"}}
"""
        try:
            raw = _call_model(model, prompt)
            data = _extract_json(raw)
            if not data:
                raise ValueError(f"Could not parse JSON from action: {raw[:300]}")
            action_val = str(data.get("action", "STOP"))
            if action_val not in ALLOWED_TOOLS:
                raise ValueError(f"Model returned forbidden action: {action_val!r}")
            return AgentAction(
                action=action_val,
                arguments=data.get("arguments", {}),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            _log.warning(f"Action generation error, defaulting to STOP: {exc}")
            return AgentAction(action="STOP", arguments={}, reason=f"Action generation failed: {exc}")

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    def verify_answerability(self, state: AgentState) -> VerificationResult:
        evidence_ctx = _build_evidence_context(state)
        contradictions_str = ""
        if state.contradictions:
            contradictions_str = "\n".join(
                f"- Claim A: {c.claim_a} (source: {c.source_a}) vs Claim B: {c.claim_b} (source: {c.source_b}), resolved: {c.resolved}"
                for c in state.contradictions
            )
        else:
            contradictions_str = "(none detected)"

        prompt = f"""{_security.build_system_boundary_prompt()}

You are a verification agent. Decide whether the evidence is sufficient to answer the question.

Question: {state.question}

Evidence retrieved:
{evidence_ctx}

Contradictions detected:
{contradictions_str}

Calls used: {state.calls_used}/6

Rules:
1. ANSWERABLE only if ALL claims in the answer are grounded in retrieved evidence.
2. INSUFFICIENT if evidence is missing, contradictory without resolution, or budget exhausted without enough data.
3. Be CONSERVATIVE. When in doubt, return INSUFFICIENT.

Output ONLY valid JSON:
{{
  "status": "ANSWERABLE" or "INSUFFICIENT",
  "supported_claims": ["<claim>"],
  "unresolved_conflicts": ["<conflict>"],
  "reason": "<explanation>"
}}
"""
        try:
            raw = _call_model(self._tier2_model, prompt)
            data = _extract_json(raw)
            if not data:
                raise ValueError(f"Could not parse verification JSON: {raw[:300]}")
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
            _log.warning(f"Verification error, defaulting to INSUFFICIENT: {exc}")
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                reason=f"Verification failed: {exc}",
            )

    # -----------------------------------------------------------------------
    # Final answer
    # -----------------------------------------------------------------------

    def generate_final_answer(self, state: AgentState) -> FinalAnswer:
        evidence_ctx = _build_evidence_context(state)
        verification = self.verify_answerability(state)

        if verification.status == VerificationStatus.INSUFFICIENT:
            conflicts = "; ".join(verification.unresolved_conflicts) if verification.unresolved_conflicts else ""
            reason_detail = f" {conflicts}" if conflicts else f" {verification.reason}"
            return FinalAnswer(
                answer=f"Insufficient information.{reason_detail}".strip(),
                sources=[],
                verification=verification,
                insufficient=True,
            )

        pages_cited = [str(p) for p in state.retrieved_pages]

        prompt = f"""{_security.build_system_boundary_prompt()}

You are generating the final answer. Use ONLY the evidence below. Do NOT make additional tool calls. Do NOT hallucinate.

Question: {state.question}

Retrieved evidence:
{evidence_ctx}

Pages retrieved: {pages_cited}

Rules:
- Answer concisely and directly.
- Cite only page numbers that were actually retrieved and contain the relevant information.
- If evidence is contradictory: explain which source supersedes the other and why, if clear from the evidence. Otherwise state "Insufficient information."
- Do NOT invent facts or page numbers.
- End with "Sources: Page X, Page Y, ..." listing only pages that contained relevant evidence.

Output ONLY valid JSON:
{{
  "answer": "<your answer text including Sources line>",
  "sources": ["Page X", "Page Y"]
}}
"""
        try:
            raw = _call_model(self._tier2_model, prompt, temperature=0.0)
            data = _extract_json(raw)
            if not data:
                raise ValueError(f"Could not parse final answer JSON: {raw[:300]}")
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
