from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DocumentToolProvider(ABC):
    """
    Abstract interface for document tools.
    The agent sees ONLY this interface — never the underlying PDF or parser.
    """

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata only: doc_id, title, pages."""

    @abstractmethod
    def list_headings(self, doc_id: str) -> List[str]:
        """Returns heading strings only."""

    @abstractmethod
    def search_keyword(self, doc_id: str, keyword: str) -> List[int]:
        """Returns page numbers only. No page text."""

    @abstractmethod
    def get_page(self, doc_id: str, page_number: int) -> Dict[str, Any]:
        """Returns exactly one page: {doc_id, page_number, text}."""
