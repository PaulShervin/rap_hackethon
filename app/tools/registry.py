from typing import Any, Dict
from app.tools.base import DocumentToolProvider


class ToolRegistry:
    """Routes validated tool calls to the DocumentToolProvider. No LLM inside."""

    def __init__(self, provider: DocumentToolProvider):
        self._provider = provider

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name == "list_documents":
            return self._provider.list_documents()

        if tool_name == "list_headings":
            return self._provider.list_headings(arguments["doc_id"])

        if tool_name == "search_keyword":
            return self._provider.search_keyword(arguments["doc_id"], arguments["keyword"])

        if tool_name == "get_page":
            page_number = int(arguments["page_number"])
            return self._provider.get_page(arguments["doc_id"], page_number)

        raise ValueError(f"Unknown tool: {tool_name}")
