"""
Deterministic pre-verification checks run BEFORE calling the LLM verifier.
No LLM here — pure logic applied to AgentState.
"""
from app.schemas.state import AgentState, AgentStatus
from app.schemas.responses import VerificationResult, VerificationStatus


class DeterministicVerifier:
    def check(self, state: AgentState) -> VerificationResult:
        conflicts = []

        # 1. Budget exhausted with no evidence
        if state.calls_used >= 6 and not state.evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                reason="Budget exhausted without retrieving any evidence.",
            )

        # 2. No pages retrieved (no get_page calls)
        page_evidence = [e for e in state.evidence if e.tool == "get_page"]
        if not page_evidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                reason="No document pages were retrieved. Cannot ground an answer.",
            )

        # 3. Unresolved contradictions
        unresolved = [c for c in state.contradictions if not c.resolved]
        if unresolved:
            for c in unresolved:
                conflicts.append(f"{c.claim_a!r} (source: {c.source_a}) conflicts with {c.claim_b!r} (source: {c.source_b})")
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT,
                supported_claims=[],
                unresolved_conflicts=conflicts,
                reason="Contradictory evidence found without resolution.",
            )

        # 4. Passes deterministic checks — let LLM verifier make final call
        return VerificationResult(
            status=VerificationStatus.ANSWERABLE,
            reason="Passed deterministic checks. LLM verification required.",
        )
