from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class LayaDecisionType(str, Enum):
    HANDLE = "HANDLE"
    ESCALATE = "ESCALATE"


class LayaDecision(BaseModel):
    decision: LayaDecisionType
    confidence: float
    reason: str
    complexity_signals: List[str] = Field(default_factory=list)


class VerificationStatus(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    INSUFFICIENT = "INSUFFICIENT"


class VerificationResult(BaseModel):
    status: VerificationStatus
    supported_claims: List[str] = Field(default_factory=list)
    unresolved_conflicts: List[str] = Field(default_factory=list)
    reason: str


class FinalAnswer(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    insufficient: bool = False
