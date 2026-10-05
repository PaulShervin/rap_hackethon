"""
Local development DocumentToolProvider backed by an uploaded PDF.
Behaves identically to the production interface from the agent's perspective.
The agent never receives a reference to this object — only results from the Harness.
"""
import io
import re
from typing import Any, Dict, List, Optional

import pdfplumber

from app.tools.base import DocumentToolProvider


class LocalPDFProvider(DocumentToolProvider):
    def __init__(self):
        self._documents: Dict[str, Dict[str, Any]] = {}

    def load_pdf(self, doc_id: str, pdf_bytes: bytes, title: str = "") -> None:
        pages: List[str] = []
        headings: List[str] = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
                for line in text.splitlines():
                    stripped = line.strip()
                    if self._looks_like_heading(stripped):
                        if stripped and stripped not in headings:
                            headings.append(stripped)

        self._documents[doc_id] = {
            "doc_id": doc_id,
            "title": title or doc_id,
            "pages": pages,
            "total_pages": total_pages,
            "headings": headings,
        }

    def _looks_like_heading(self, line: str) -> bool:
        if not line:
            return False
        if len(line) > 120:
            return False
        if line.endswith(".") and len(line) > 60:
            return False
        patterns = [
            r"^\d+\.\s+[A-Z]",
            r"^[A-Z][A-Z\s]{4,}$",
            r"^(Chapter|Section|Article|Part)\s+\d",
            r"^[IVX]+\.\s+[A-Z]",
        ]
        return any(re.match(p, line) for p in patterns)

    def remove_document(self, doc_id: str) -> None:
        self._documents.pop(doc_id, None)

    def list_documents(self) -> List[Dict[str, Any]]:
        return [
            {"doc_id": d["doc_id"], "title": d["title"], "pages": d["total_pages"]}
            for d in self._documents.values()
        ]

    def list_headings(self, doc_id: str) -> List[str]:
        doc = self._get_doc(doc_id)
        return doc["headings"]

    def search_keyword(self, doc_id: str, keyword: str) -> List[int]:
        doc = self._get_doc(doc_id)
        keyword_lower = keyword.lower().strip()
        results: List[int] = []
        for i, page_text in enumerate(doc["pages"], start=1):
            if keyword_lower in page_text.lower():
                results.append(i)
        return results

    def get_page(self, doc_id: str, page_number: int) -> Dict[str, Any]:
        doc = self._get_doc(doc_id)
        pages = doc["pages"]
        if page_number < 1 or page_number > len(pages):
            raise ValueError(f"Page {page_number} does not exist in '{doc_id}' (total: {len(pages)}).")
        return {
            "doc_id": doc_id,
            "page_number": page_number,
            "text": pages[page_number - 1],
        }

    def _get_doc(self, doc_id: str) -> Dict[str, Any]:
        doc = self._documents.get(doc_id)
        if not doc:
            raise ValueError(f"Document '{doc_id}' not found. Available: {list(self._documents.keys())}")
        return doc

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._documents
