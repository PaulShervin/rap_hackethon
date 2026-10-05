import pytest
from app.tools.base import DocumentToolProvider
from app.tools.registry import ToolRegistry
from app.harness.harness import Harness
from app.schemas.state import AgentState
from typing import Dict, Any, List


class MockDocumentProvider(DocumentToolProvider):
    """In-memory document provider for testing."""

    def __init__(self):
        self._pages: Dict[str, List[str]] = {}
        self._headings: Dict[str, List[str]] = {}

    def add_document(self, doc_id: str, pages: List[str], headings: List[str] = None):
        self._pages[doc_id] = pages
        self._headings[doc_id] = headings or []

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {"doc_id": doc_id, "title": doc_id, "pages": len(pages)}
            for doc_id, pages in self._pages.items()
        ]

    def list_headings(self, doc_id: str) -> List[str]:
        return self._headings.get(doc_id, [])

    def search_keyword(self, doc_id: str, keyword: str) -> List[int]:
        pages = self._pages.get(doc_id, [])
        return [i + 1 for i, text in enumerate(pages) if keyword.lower() in text.lower()]

    def get_page(self, doc_id: str, page_number: int) -> Dict[str, Any]:
        pages = self._pages.get(doc_id, [])
        if page_number < 1 or page_number > len(pages):
            raise ValueError(f"Page {page_number} out of range for {doc_id}.")
        return {"doc_id": doc_id, "page_number": page_number, "text": pages[page_number - 1]}


@pytest.fixture
def mock_provider():
    return MockDocumentProvider()


@pytest.fixture
def simple_provider():
    p = MockDocumentProvider()
    p.add_document(
        "doc_001",
        pages=[
            "Introduction to Employee Policy.",
            "Leave Policy: The maximum leave period is 30 days per year.",
            "Cancellation Policy: The cancellation period is 30 days.",
            "Effective January 2026: cancellation period is 15 days.",
            "This policy supersedes the previous cancellation policy on page 3.",
        ],
        headings=["Introduction", "Leave Policy", "Cancellation Policy"],
    )
    return p


@pytest.fixture
def harness(simple_provider):
    registry = ToolRegistry(simple_provider)
    return Harness(registry)


@pytest.fixture
def fresh_state():
    return AgentState(
        question_id="Q_test",
        question="What is the cancellation policy?",
        document_id="doc_001",
        calls_remaining=6,
    )
