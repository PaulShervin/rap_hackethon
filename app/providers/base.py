from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.schemas.responses import LayaDecision, VerificationResult, FinalAnswer
from app.schemas.state import AgentState
from app.schemas.actions import AgentAction


class LLMProvider(ABC):

    @abstractmethod
    def generate_laya_decision(self, question: str, doc_metadata: List[Dict]) -> LayaDecision:
        """Tier 1: Decide HANDLE or ESCALATE."""

    @abstractmethod
    def generate_laya_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        """Tier 1: Choose the next document tool action."""

    @abstractmethod
    def generate_tier2_action(self, state: AgentState, doc_metadata: List[Dict]) -> AgentAction:
        """Tier 2: Choose the next document tool action or STOP."""

    @abstractmethod
    def generate_final_answer(self, state: AgentState) -> FinalAnswer:
        """Generate the final answer using only evidence in state."""

    @abstractmethod
    def verify_answerability(self, state: AgentState) -> VerificationResult:
        """Deterministically-guided verification of whether evidence is sufficient."""
