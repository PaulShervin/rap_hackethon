"""Tests for the document tool implementations."""
import pytest
from tests.conftest import MockDocumentProvider
from app.tools.registry import ToolRegistry


@pytest.fixture
def provider():
    p = MockDocumentProvider()
    p.add_document(
        "doc_001",
        pages=[
            "Introduction.",
            "Leave period is 20 days.",
            "Cancellation period is 30 days.",
        ],
        headings=["Introduction", "Leave Policy", "Cancellation"],
    )
    return p


def test_list_documents_returns_metadata_only(provider):
    docs = provider.list_documents()
    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"] == "doc_001"
    assert doc["pages"] == 3
    assert "text" not in doc  # must not return full text


def test_list_headings_returns_strings(provider):
    headings = provider.list_headings("doc_001")
    assert isinstance(headings, list)
    assert "Introduction" in headings


def test_search_keyword_returns_page_numbers_only(provider):
    results = provider.search_keyword("doc_001", "leave")
    assert isinstance(results, list)
    assert all(isinstance(r, int) for r in results)
    assert 2 in results


def test_search_keyword_returns_no_text(provider):
    results = provider.search_keyword("doc_001", "leave")
    for r in results:
        assert isinstance(r, int)  # only page numbers


def test_get_page_returns_single_page(provider):
    page = provider.get_page("doc_001", 2)
    assert page["page_number"] == 2
    assert "Leave period" in page["text"]
    assert page["doc_id"] == "doc_001"


def test_get_page_invalid_raises(provider):
    with pytest.raises(ValueError):
        provider.get_page("doc_001", 99)


def test_get_page_negative_raises(provider):
    with pytest.raises(ValueError):
        provider.get_page("doc_001", 0)


def test_registry_routes_correctly(provider):
    registry = ToolRegistry(provider)
    result = registry.execute("search_keyword", {"doc_id": "doc_001", "keyword": "cancellation"})
    assert isinstance(result, list)
    assert 3 in result


def test_search_case_insensitive(provider):
    results = provider.search_keyword("doc_001", "LEAVE")
    assert 2 in results
