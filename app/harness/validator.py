from typing import Tuple, Optional
from app.schemas.actions import ALLOWED_TOOLS, DOCUMENT_TOOLS


class ActionValidator:
    def validate(self, action: str, arguments: dict) -> Tuple[bool, str]:
        if not action:
            return False, "Action is empty."

        if action not in ALLOWED_TOOLS:
            return False, f"Action '{action}' is not in the allowed tool list. Forbidden tools include: download_pdf, read_pdf, semantic_search, vector_search, embedding_search, get_all_pages."

        if action == "list_documents":
            return True, ""

        if action == "list_headings":
            doc_id = arguments.get("doc_id")
            if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
                return False, "list_headings requires a non-empty string 'doc_id'."
            return True, ""

        if action == "search_keyword":
            doc_id = arguments.get("doc_id")
            keyword = arguments.get("keyword")
            if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
                return False, "search_keyword requires a non-empty string 'doc_id'."
            if not keyword or not isinstance(keyword, str) or not keyword.strip():
                return False, "search_keyword requires a non-empty string 'keyword'."
            return True, ""

        if action == "get_page":
            doc_id = arguments.get("doc_id")
            page_number = arguments.get("page_number")
            if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
                return False, "get_page requires a non-empty string 'doc_id'."
            if page_number is None:
                return False, "get_page requires 'page_number'."
            if not isinstance(page_number, int):
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    return False, f"get_page 'page_number' must be an integer, got: {page_number!r}"
            if page_number < 1:
                return False, f"get_page 'page_number' must be >= 1, got: {page_number}"
            if "page_range" in arguments or "pages" in arguments:
                return False, "Page ranges are not allowed. Request one page at a time."
            return True, ""

        if action == "STOP":
            return True, ""

        return False, f"Unhandled action: {action}"

    def is_document_tool(self, action: str) -> bool:
        return action in DOCUMENT_TOOLS
