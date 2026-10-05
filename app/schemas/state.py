from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from enum import Enum
import uuid


class Tier(str, Enum):
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"


class AgentStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INSUFFICIENT = "INSUFFICIENT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ERROR = "ERROR"


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool: str
    arguments: Dict[str, Any]
    result: Any
    call_number: int


class Contradiction(BaseModel):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    resolved: bool = False
    resolution: Optional[str] = None


class AgentState(BaseModel):
    question_id: str
    question: str
    document_id: Optional[str] = None

    current_tier: Tier = Tier.TIER_1

    calls_used: int = 0
    calls_remaining: int = 6

    evidence: List[Evidence] = Field(default_factory=list)
    searched_keywords: List[str] = Field(default_factory=list)
    retrieved_pages: List[int] = Field(default_factory=list)

    known_facts: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)

    contradictions: List[Contradiction] = Field(default_factory=list)
    supersession_candidates: List[str] = Field(default_factory=list)

    last_action: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[Any] = None

    status: AgentStatus = AgentStatus.RUNNING

    escalation_reason: Optional[str] = None
    answerability: Optional[str] = None

    def get_evidence_for_page(self, doc_id: str, page_number: int) -> Optional[Evidence]:
        for ev in self.evidence:
            if ev.tool == "get_page" and ev.arguments.get("doc_id") == doc_id and ev.arguments.get("page_number") == page_number:
                return ev
        return None

    def get_evidence_for_keyword(self, doc_id: str, keyword: str) -> Optional[Evidence]:
        for ev in self.evidence:
            if ev.tool == "search_keyword" and ev.arguments.get("doc_id") == doc_id and ev.arguments.get("keyword", "").lower() == keyword.lower():
                return ev
        return None

    def get_all_page_texts(self) -> List[str]:
        texts = []
        for ev in self.evidence:
            if ev.tool == "get_page" and isinstance(ev.result, dict):
                texts.append(ev.result.get("text", ""))
        return texts

    def summary_for_llm(self) -> str:
        parts = [
            f"Question: {self.question}",
            f"Calls used: {self.calls_used}/6",
            f"Calls remaining: {self.calls_remaining}",
            f"Tier: {self.current_tier.value}",
        ]
        if self.searched_keywords:
            parts.append(f"Keywords searched: {self.searched_keywords}")
        if self.retrieved_pages:
            parts.append(f"Pages retrieved: {self.retrieved_pages}")
        if self.known_facts:
            parts.append(f"Known facts: {self.known_facts}")
        if self.contradictions:
            parts.append(f"Contradictions detected: {len(self.contradictions)}")
        if self.unresolved_questions:
            parts.append(f"Unresolved: {self.unresolved_questions}")
        return "\n".join(parts)
