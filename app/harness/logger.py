import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger("rap.harness")


class TraceEntry:
    def __init__(
        self,
        question_id: str,
        tier: str,
        call_number: int,
        action: str,
        arguments: Dict[str, Any],
        status: str,
        reason: str = "",
        result_summary: str = "",
        calls_remaining: int = 0,
        cached: bool = False,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.question_id = question_id
        self.tier = tier
        self.call_number = call_number
        self.action = action
        self.arguments = arguments
        self.status = status
        self.reason = reason
        self.result_summary = result_summary
        self.calls_remaining = calls_remaining
        self.cached = cached

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "question_id": self.question_id,
            "tier": self.tier,
            "call_number": self.call_number,
            "action": self.action,
            "arguments": self.arguments,
            "status": self.status,
            "reason": self.reason,
            "result_summary": self.result_summary,
            "calls_remaining": self.calls_remaining,
            "cached": self.cached,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class HarnessLogger:
    def __init__(self):
        self._entries: List[TraceEntry] = []

    def log(self, entry: TraceEntry):
        self._entries.append(entry)
        _log.info(json.dumps(entry.to_dict()))

    def log_success(self, question_id: str, tier: str, call_number: int, action: str,
                    arguments: dict, result_summary: str, reason: str, calls_remaining: int,
                    cached: bool = False):
        entry = TraceEntry(
            question_id=question_id, tier=tier, call_number=call_number,
            action=action, arguments=arguments, status="SUCCESS",
            reason=reason, result_summary=result_summary,
            calls_remaining=calls_remaining, cached=cached,
        )
        self.log(entry)

    def log_blocked(self, question_id: str, tier: str, call_number: int, action: str,
                    arguments: dict, reason: str, calls_remaining: int):
        entry = TraceEntry(
            question_id=question_id, tier=tier, call_number=call_number,
            action=action, arguments=arguments, status="BLOCKED",
            reason=reason, calls_remaining=calls_remaining,
        )
        self.log(entry)

    def log_rejected(self, question_id: str, tier: str, action: str,
                     arguments: dict, reason: str):
        entry = TraceEntry(
            question_id=question_id, tier=tier, call_number=-1,
            action=action, arguments=arguments, status="REJECTED",
            reason=reason,
        )
        self.log(entry)

    def get_trace(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def clear(self):
        self._entries.clear()
