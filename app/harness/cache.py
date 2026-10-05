from typing import Optional, Any
from app.schemas.state import AgentState, Evidence


class EvidenceCache:
    """Question-scoped evidence cache. Prevents redundant document tool calls."""

    def check_get_page(self, state: AgentState, doc_id: str, page_number: int) -> Optional[Evidence]:
        return state.get_evidence_for_page(doc_id, page_number)

    def check_search_keyword(self, state: AgentState, doc_id: str, keyword: str) -> Optional[Evidence]:
        return state.get_evidence_for_keyword(doc_id, keyword)

    def check_list_documents(self, state: AgentState) -> Optional[Evidence]:
        for ev in state.evidence:
            if ev.tool == "list_documents":
                return ev
        return None

    def check_list_headings(self, state: AgentState, doc_id: str) -> Optional[Evidence]:
        for ev in state.evidence:
            if ev.tool == "list_headings" and ev.arguments.get("doc_id") == doc_id:
                return ev
        return None

    def lookup(self, state: AgentState, tool: str, arguments: dict) -> Optional[Evidence]:
        if tool == "get_page":
            return self.check_get_page(state, arguments.get("doc_id", ""), arguments.get("page_number", -1))
        if tool == "search_keyword":
            return self.check_search_keyword(state, arguments.get("doc_id", ""), arguments.get("keyword", ""))
        if tool == "list_documents":
            return self.check_list_documents(state)
        if tool == "list_headings":
            return self.check_list_headings(state, arguments.get("doc_id", ""))
        return None
