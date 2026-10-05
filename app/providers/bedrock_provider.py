"""
Amazon Bedrock Provider — handles both Tier 1 (Laya) and Tier 2 (reasoning).
Uses Claude Sonnet 4.5 via bedrock converse() API.

Environment variables:
    BEDROCK_MODEL   — model ID (default: anthropic.claude-sonnet-4-5-20250929-v1:0)
    BEDROCK_REGION  — AWS region  (default: us-west-2, also reads AWS_REGION)

AWS credentials come from the ambient boto3 credential chain
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN).
No credentials are stored here.
"""
import logging
from typing import Dict, List

from app.providers.base import LLMProvider
from app.providers.bedrock_client import call_bedrock, check_bedrock_available, extract_json
from app.providers.local_laya_provider import _heuristic_decision, _build_light_evidence
from app.schemas.actions import AgentAction, ALLOWED_TOOLS
from app.schemas.responses import (
    FinalAnswer, LayaDecision, LayaDecisionType,
    VerificationResult, VerificationStatus,
)
from app.schemas.state import AgentState
from app.harness.security import SecurityPolicy

_log = logging.getLogger("rap.bedrock_provider")
_security = SecurityPolicy()


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
            parts.append(f"[list_headings({ev.arguments.get('doc_id')!r})] → {ev.result}")
    return "\n\n".join(parts) if parts else "(no evidence retrieved yet)"


class BedrockProvider(LLMProvider):
    """Single provider for both Tier 1 and Tier 2 using Bedrock Claude Sonnet 4.5."""

    def __init__(self):
        ok, msg = check_bedrock_available()
        if not ok:
            _log.error(
                f"Bedrock availability check failed: {msg}\n"
                "Ensure AWS credentials are set and the model is enabled in us-west-2."
            )
        else:
            _log.info(f"Bedrock provider ready: {msg}")
        self._available = ok

    # ------------------------------------------------------------------
    # Tier 1 — Laya decision
    # ------------------------------------------------------------------

    def generate_laya_decision(self, question: str, doc_metadata: List[Dict]) -> LayaDecision:
        if not self._available:
            _log.warning("Bedrock unavailable — using heuristic Laya decision.")
            return _heuristic_decision(question)

        prompt = f"""{_security.build_system_boundary_prompt()}

You are Laya, a first-level decision agent for a document-answering system.

Analyze the question and decide HANDLE (simple direct factual lookup, ≤2-3 tool calls) or ESCALATE (complex multi-step reasoning required).

Documents available: {doc_metadata}
Question: {question}

ESCALATE when you detect:
- Conflicting statements on different pages
- Temporal reasoning (effective dates, revisions, supersession)
- Which of two policies currently applies
- Multiple conditions or exceptions
- Policy interpretation or nuance
- Estimated evidence > 3 pages

HANDLE only for simple direct lookups where the answer is a single fact.

Respond with ONLY this JSON:
{{
  "decision": "HANDLE" or "ESCALATE",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "complexity_signals": ["<signal if any>"]
}}"""
        try:
            raw = call_bedrock(prompt)
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
            _log.warning(f"Bedrock Laya decision error: {exc}. Using heuristic fallback.")
            return _heuristic_decision(question)
        except Exception as exc:
            _log.warning(f"Laya decision parse error: {exc}. Using heuristic fallback.")
            return _heuristic_decision(question)

    # ------------------------------------------------------------------
    # Tier 1 — Laya action
    # ------------------------------------------------------------------

    def generate_laya_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        if not self._available:
            return AgentAction(action="STOP", arguments={}, reason="Bedrock unavailable.")

        evidence_ctx = _build_light_evidence(state)
        prompt = f"""{_security.build_system_boundary_prompt()}

You are Laya, handling a simple document question. Choose the next tool call.

ALLOWED TOOLS ONLY:
- list_documents()  → {{}}
- list_headings(doc_id) → {{"doc_id": "..."}}
- search_keyword(doc_id, keyword) → {{"doc_id": "...", "keyword": "..."}} returns page numbers only
- get_page(doc_id, page_number) → {{"doc_id": "...", "page_number": <int>}} returns one page

Question: {state.question}
Calls used: {state.calls_used}/6 | Remaining: {state.calls_remaining}
Keywords already searched: {state.searched_keywords}
Pages already retrieved: {state.retrieved_pages}
Evidence so far: {evidence_ctx}

If you have enough evidence or cannot make progress, output STOP.

Respond with ONLY this JSON:
{{
  "action": "<list_documents|list_headings|search_keyword|get_page|STOP>",
  "arguments": {{}},
  "reason": "<why>"
}}"""
        try:
            raw = call_bedrock(prompt)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in Laya action response: {raw[:200]}")
            action_val = str(data.get("action", "STOP"))
            if action_val not in ALLOWED_TOOLS:
                raise ValueError(f"Forbidden action: {action_val!r}")
            return AgentAction(
                action=action_val,
                arguments=data.get("arguments", {}),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            _log.warning(f"Laya action error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Error: {exc}")

    # ------------------------------------------------------------------
    # Tier 2 — action
    # ------------------------------------------------------------------

    def generate_tier2_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        if not self._available:
            return AgentAction(action="STOP", arguments={}, reason="Bedrock unavailable.")

        evidence_ctx = _build_evidence_context(state)
        prompt = f"""{_security.build_system_boundary_prompt()}

You are a document-answering reasoning agent. Answer the user's question using ONLY the four allowed document tools.

ALLOWED TOOLS:
1. list_documents()
2. list_headings(doc_id)
3. search_keyword(doc_id, keyword)   — returns PAGE NUMBERS ONLY
4. get_page(doc_id, page_number)     — returns ONE page of text

FORBIDDEN: semantic_search, vector_search, download_pdf, read_pdf, get_all_pages, or any other tool.

Current state:
{state.summary_for_llm()}

Documents: {doc_metadata}

Evidence retrieved so far:
{evidence_ctx}

RULES:
- You have {state.calls_remaining} tool calls remaining. Budget wisely.
- Do NOT repeat searches/page retrievals already done.
- If you have enough evidence to answer, output STOP.
- If evidence is contradictory, look for a page that clarifies supersession.
- If budget will run out without enough evidence, output STOP.

Respond with ONLY this JSON:
{{
  "action": "<list_documents|list_headings|search_keyword|get_page|STOP>",
  "arguments": {{}},
  "reason": "<why this action>"
}}

For search_keyword: {{"doc_id": "...", "keyword": "..."}}
For get_page: {{"doc_id": "...", "page_number": <integer>}}
For list_headings: {{"doc_id": "..."}}
For STOP or list_documents: {{}}"""
        try:
            raw = call_bedrock(prompt)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in Tier 2 action response: {raw[:300]}")
            action_val = str(data.get("action", "STOP"))
            if action_val not in ALLOWED_TOOLS:
                raise ValueError(f"Forbidden action: {action_val!r}")
            return AgentAction(
                action=action_val,
                arguments=data.get("arguments", {}),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            _log.warning(f"Tier 2 action error: {exc}. Defaulting to STOP.")
            return AgentAction(action="STOP", arguments={}, reason=f"Error: {exc}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_answerability(self, state: AgentState) -> VerificationResult:
        if not self._available:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                reason="Bedrock unavailable.",
            )

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

You are a verification agent. Decide whether retrieved evidence is sufficient to answer the question.

Question: {state.question}

Evidence:
{evidence_ctx}

Contradictions:
{contradictions_str}

Calls used: {state.calls_used}/6

RULES:
1. ANSWERABLE only if ALL claims are directly supported by the evidence above.
2. INSUFFICIENT if evidence is missing, contradictory without resolution, or budget exhausted.
3. Be CONSERVATIVE. When uncertain, return INSUFFICIENT.
4. Do NOT hallucinate facts not in the evidence.

Respond with ONLY this JSON:
{{
  "status": "ANSWERABLE" or "INSUFFICIENT",
  "supported_claims": ["<claim>"],
  "unresolved_conflicts": ["<conflict if any>"],
  "reason": "<explanation>"
}}"""
        try:
            raw = call_bedrock(prompt, temperature=0.0)
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

    # ------------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------------

    def generate_final_answer(self, state: AgentState) -> FinalAnswer:
        if not self._available:
            return FinalAnswer(
                answer="Insufficient information. Bedrock is unavailable.",
                sources=[],
                insufficient=True,
            )

        evidence_ctx = _build_evidence_context(state)
        verification = self.verify_answerability(state)

        if verification.status == VerificationStatus.INSUFFICIENT:
            conflicts = (
                "; ".join(verification.unresolved_conflicts)
                if verification.unresolved_conflicts
                else verification.reason
            )
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
- End your answer with: Sources: Page X, Page Y

Respond with ONLY this JSON:
{{
  "answer": "<complete answer text including Sources line>",
  "sources": ["Page X", "Page Y"]
}}"""
        try:
            raw = call_bedrock(prompt, temperature=0.0)
            data = extract_json(raw)
            if not data:
                raise ValueError(f"No JSON in final answer: {raw[:300]}")
            return FinalAnswer(
                answer=str(data.get("answer", "Insufficient information.")),
                sources=data.get("sources", []),
                verification=verification,
                insufficient=False,
            )
        except Exception as exc:
            _log.warning(f"Final answer error: {exc}")
            return FinalAnswer(
                answer="Insufficient information. Answer generation encountered an error.",
                sources=[],
                verification=verification,
                insufficient=True,
            )
