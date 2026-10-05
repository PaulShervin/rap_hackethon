from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any
from enum import Enum


class ToolName(str, Enum):
    LIST_DOCUMENTS = "list_documents"
    LIST_HEADINGS = "list_headings"
    SEARCH_KEYWORD = "search_keyword"
    GET_PAGE = "get_page"
    STOP = "STOP"


ALLOWED_TOOLS = {t.value for t in ToolName}

DOCUMENT_TOOLS = {
    ToolName.LIST_DOCUMENTS.value,
    ToolName.LIST_HEADINGS.value,
    ToolName.SEARCH_KEYWORD.value,
    ToolName.GET_PAGE.value,
}


class AgentAction(BaseModel):
    action: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("action")
    @classmethod
    def action_must_be_allowed(cls, v):
        if v not in ALLOWED_TOOLS:
            raise ValueError(f"Action '{v}' is not in the allowed tool list: {ALLOWED_TOOLS}")
        return v
