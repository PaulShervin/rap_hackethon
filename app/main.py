"""
Application entry point and orchestrator.

Architecture:
    PDF upload → DocumentToolProvider
    Question → Laya (Tier 1, local Ollama) → HANDLE or ESCALATE
                                              ↓
                              Gemma 3 (Tier 2, local Ollama)
                                              ↓
                           Shared Deterministic Harness
                                              ↓
                              Restricted Document Tools
                                              ↓
                        Question-Scoped Evidence Cache
                                              ↓
                        Verification → Grounded Answer
"""
import logging
import os
import uuid
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

load_dotenv()

from app.tools.document_tools import LocalPDFProvider
from app.tools.registry import ToolRegistry
from app.harness.harness import Harness
from app.agents.laya_agent import LayaAgent
from app.agents.tier2_agent import Tier2Agent
from app.providers.bedrock_provider import BedrockProvider
from app.schemas.state import AgentState
from app.harness.budget import MAX_TOOL_CALLS

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_log = logging.getLogger("rap.main")


class DocumentAnsweringApp:
    def __init__(self):
        self.pdf_provider = LocalPDFProvider()
        self.tool_registry = ToolRegistry(self.pdf_provider)
        self.harness = Harness(self.tool_registry)

        # Single Bedrock provider serves both Tier 1 (Laya) and Tier 2
        self.laya_provider = BedrockProvider()
        self.tier2_provider = self.laya_provider  # same instance, implements all 5 methods
        self.laya = LayaAgent(self.harness, self.laya_provider)
        self.tier2 = Tier2Agent(self.harness, self.tier2_provider)

        self._active_doc_id: Optional[str] = None
        self._last_trace: List[Dict] = []
        self._last_state: Optional[AgentState] = None
        self._last_laya_decision = None

    def upload_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        title = filename.replace(".pdf", "").replace("_", " ").title()
        self.pdf_provider.load_pdf(doc_id, pdf_bytes, title)
        self._active_doc_id = doc_id
        _log.info(f"Loaded PDF: doc_id={doc_id!r} title={title!r}")
        return doc_id

    def remove_pdf(self, doc_id: str):
        self.pdf_provider.remove_document(doc_id)
        if self._active_doc_id == doc_id:
            self._active_doc_id = None

    def ask(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        target_doc_id = doc_id or self._active_doc_id
        if not target_doc_id:
            return {
                "answer": "No document uploaded. Please upload a PDF first.",
                "trace": [],
                "insufficient": True,
                "calls_used": 0,
            }

        question_id = f"Q_{uuid.uuid4().hex[:6]}"
        self.harness.reset_trace()

        state = AgentState(
            question_id=question_id,
            question=question,
            document_id=target_doc_id,
            calls_remaining=MAX_TOOL_CALLS,
        )

        doc_metadata = self.pdf_provider.list_documents()

        # Phase 1: Laya (Tier 1)
        state, laya_decision, should_escalate = self.laya.run(state, doc_metadata)
        self._last_laya_decision = laya_decision

        # Phase 2: Gemma (Tier 2) — runs on escalation OR when Laya has remaining budget
        if should_escalate or state.calls_remaining > 0:
            _log.info(
                f"Running Tier 2 (Gemma). Escalated={should_escalate}, "
                f"calls_used={state.calls_used}, calls_remaining={state.calls_remaining}"
            )
            answer = self.tier2.run(state, doc_metadata)
        else:
            # Laya used all budget and chose to stop — generate final answer via Tier 2 provider
            answer = self.tier2_provider.generate_final_answer(state)

        self._last_trace = self.harness.get_trace()
        self._last_state = state

        return {
            "question_id": question_id,
            "question": question,
            "answer": answer.answer,
            "sources": answer.sources,
            "insufficient": answer.insufficient,
            "calls_used": state.calls_used,
            "calls_max": MAX_TOOL_CALLS,
            "tier": state.current_tier.value,
            "laya_decision": laya_decision.model_dump() if laya_decision else None,
            "escalated": should_escalate,
            "trace": self._last_trace,
            "verification": answer.verification.model_dump() if answer.verification else None,
        }

    def get_documents(self) -> List[Dict]:
        return self.pdf_provider.list_documents()

    def get_active_doc_id(self) -> Optional[str]:
        return self._active_doc_id

    def health_check(self) -> Dict[str, Any]:
        """Check Bedrock availability."""
        from app.providers.bedrock_client import check_bedrock_available, _model_id, _region
        ok, msg = check_bedrock_available()
        info = {"available": ok, "model": _model_id(), "region": _region(), "message": msg}
        return {"laya": info, "tier2": info}
