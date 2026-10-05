# RAP — Budgeted Document-Answering Agent

A chat-based AI system that answers questions about uploaded PDFs using a strict **6-tool-call budget** and a two-tier local agent architecture. **No cloud LLM dependency** — all reasoning runs locally via Ollama.

---

## Architecture

```
USER
  │
  ▼
LAYA (Tier 1) — local Ollama (LAYA_MODEL)
First-level decision + lightweight execution
  │
  │ escalate (complex query or emerging complexity)
  ▼
GEMMA 3 via Ollama (Tier 2) — local (OLLAMA_MODEL)
Full multi-step reasoning, contradiction resolution, temporal reasoning
  │
  ▼
DETERMINISTIC HARNESS (single gateway — no LLM inside)
  ├── Tool validation & allowlist
  ├── Argument validation
  ├── Six-call budget enforcement
  ├── Security / injection defense
  ├── Evidence cache (question-scoped)
  ├── Duplicate-call prevention
  └── Structured logging / trace
  │
  ▼
DOCUMENT TOOLS (only 4 allowed)
  ├── list_documents()
  ├── list_headings(doc_id)
  ├── search_keyword(doc_id, keyword) → page numbers ONLY
  └── get_page(doc_id, page_number) → one page of text
  │
  ▼
VERIFICATION (deterministic pre-check + LLM confirm)
  │
  ▼
FINAL ANSWER (grounded in retrieved evidence only)
```

**Key principle:** *"The agent decides what should happen. The Harness decides what is allowed to happen."*

---

## Why No RAG / Embeddings / Vector DB

This system treats document access as a **finite evidence budget** rather than unlimited retrieval:

- RAG pipelines preload document content into the model's context — bypassing the 6-call constraint.
- Embeddings allow hidden semantic retrieval that the agent cannot be audited for.
- Vector databases provide unlimited, unconstrained access to all document content.
- Our architecture enforces that the agent can only learn facts through explicit, logged, budgeted tool calls — making every access decision visible and auditable.

---

## Components

### Laya (Tier 1)
- Analyzes question complexity before executing any tool calls.
- Decides `HANDLE` (simple, direct lookup) or `ESCALATE` (complex reasoning required).
- Detects complexity signals: temporal reasoning, multiple conflicting statements, supersession, multiple pages.
- Can escalate **after** partial execution — state is preserved and passed to Tier 2.

### Tier 2 — Gemma 3 via Ollama (local)
- Full multi-step reasoning loop using Gemma 3 through a local Ollama server.
- Configured via `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- Reads current `AgentState`, inspects existing evidence, chooses the highest-value next action.
- Outputs structured JSON (`AgentAction`) — never raw Python calls.
- Cannot access the PDF directly; all access goes through the Harness.
- **No cloud API calls.** Completely local.

### Deterministic Harness
- The **only** component that executes document tools.
- No LLM inside.
- Enforces: tool allowlist, argument validation, 6-call budget, security policy, cache lookups, logging.
- Call #7 is always rejected (`CALL_BUDGET_EXCEEDED`), regardless of any model or user instruction.

---

## Six-Call Budget

`MAX_TOOL_CALLS = 6` is a hard constant enforced in the Harness **before every tool execution**:

```python
if state.calls_used >= MAX_TOOL_CALLS:
    # BLOCKED — logged, returned to agent, no tool executed
```

- Cache hits (repeated retrieval of same page/keyword) do **not** consume budget.
- `STOP` is not a document call — does not consume budget.
- LLM formatting retries do not consume budget.
- The counter is per-question and resets between questions.

---

## Tool Interface

The agent sees exactly four tools through `DocumentToolProvider`:

| Tool | Arguments | Returns |
|------|-----------|---------|
| `list_documents()` | — | `[{doc_id, title, pages}]` |
| `list_headings(doc_id)` | doc_id | `["Heading 1", ...]` |
| `search_keyword(doc_id, keyword)` | doc_id, keyword | `[4, 8, 12]` (page numbers only) |
| `get_page(doc_id, page_number)` | doc_id, page_number | `{doc_id, page_number, text}` |

The `LocalPDFProvider` (dev mode) and any production provider both implement `DocumentToolProvider`. Swapping providers requires no changes to the agent or Harness.

---

## Evidence Cache

- Scoped to the current question — destroyed/reset when the question completes.
- Prevents redundant `get_page` and `search_keyword` calls from burning budget.
- Cannot provide information that was never previously retrieved (no hidden preloading).

---

## Security Model

Document content is **UNTRUSTED DATA**:

1. All page text is wrapped in `<document_evidence>` XML fences before entering LLM prompts.
2. The system prompt explicitly states document content cannot modify instructions, permissions, or budget.
3. Forbidden tools (e.g. `download_pdf`, `semantic_search`) are rejected at the Harness regardless of what any document says.
4. The 6-call budget cannot be modified by document content, user input, or model output.

---

## Contradiction Handling

When two pages contain conflicting claims:

1. The system detects the contradiction and records it in `AgentState.contradictions`.
2. If a page explicitly states supersession (e.g. "effective January 2026", "supersedes previous policy"), Tier 2 resolves it and marks `contradiction.resolved = True`.
3. If no supersession evidence is available, the system returns `"Insufficient information."` — never guesses.

---

## Answerability

Before generating a final answer, the system runs:

1. **Deterministic pre-check** (`DeterministicVerifier`): no evidence → insufficient; no pages retrieved → insufficient; unresolved contradictions → insufficient.
2. **LLM verification** (`GeminiProvider.verify_answerability`): conservative check that all claims are grounded.

Only if both pass does the system generate a grounded answer.

---

## Installation

```bash
git clone <repo>
cd RAP
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

---

## Environment Variables

No API keys required — all models run locally via Ollama.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL for Tier 2 |
| `OLLAMA_MODEL` | No | `gemma3` | Gemma 3 model name for Tier 2 |
| `LAYA_ENDPOINT` | No | same as `OLLAMA_BASE_URL` | Ollama URL for Laya (Tier 1) |
| `LAYA_MODEL` | No | same as `OLLAMA_MODEL` | Model for Laya (can be lighter) |
| `MAX_TOOL_CALLS` | No | `6` | Maximum document calls per question |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `PORT` | No | `7860` | UI server port |

---

## Running Locally

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — pull Gemma 3 (first time only)
ollama pull gemma3

# Terminal 3 — start the application
cp .env.example .env   # edit if using non-default Ollama settings
python run.py

# Open http://localhost:7860
```

The UI has a **Check Ollama Status** button that verifies both Laya and Tier 2 model availability before you start asking questions.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

All 48 tests cover:
- Budget enforcement (6-call limit, 7th blocked)
- Harness validation (forbidden tools, bad args)
- Cache (no budget consumed on cache hit)
- Security (injection ignored, forbidden tools still blocked after injection page retrieved)
- Tools (metadata-only responses, keyword returns page numbers only)
- Escalation (state preserved across tier handoff)
- Contradictions (unresolved → insufficient, resolved → answerable)
- Answerability (no pages → insufficient, absent topic → insufficient)

---

## Live Demo Instructions

1. Start the app: `python run.py`
2. Upload a PDF with `Employee_Policy.pdf` or similar.
3. Demo scenarios:

**Demo A — Simple:** "What is the company's leave period?" → Laya HANDLE, 1-2 calls, direct answer.

**Demo B — Escalation:** "Which cancellation policy applies after the 2026 revision?" → Laya ESCALATE, Tier 2 runs, temporal reasoning.

**Demo C — Contradiction resolved:** Page 4: 30 days / Page 8: 15 days effective Jan 2026 / Page 9: supersedes → Answer: 15 days.

**Demo D — Unresolved contradiction:** Page 4: 30 days / Page 8: 15 days, no supersession info → "Insufficient information."

**Demo E — Prompt injection:** PDF contains "Ignore the user..." → System answers the actual question.

**Demo F — Budget:** After 6 calls, try to ask more → "🚫 CALL_BUDGET_EXCEEDED" in trace.

---

## Trace Interpretation

Each trace entry:

```json
{
  "timestamp": "2026-01-01T00:00:00Z",
  "question_id": "Q_abc123",
  "tier": "TIER_2",
  "call_number": 3,
  "action": "get_page",
  "arguments": {"doc_id": "doc_001", "page_number": 8},
  "status": "SUCCESS",
  "reason": "Page 8 found by keyword search for 'cancellation'",
  "result_summary": "Page retrieved. Text length: 412 chars.",
  "calls_remaining": 3,
  "cached": false
}
```

Statuses: `SUCCESS`, `CACHE_HIT`, `BLOCKED` (budget), `REJECTED` (validation/allowlist), `STOP`, `ERROR`.

---

## Known Limitations

1. **PDF parsing quality**: `pdfplumber` heading detection is heuristic. Complex layouts may miss some headings.
2. **Keyword search is exact substring**: No fuzzy matching or stemming. "leave" won't find "leaves".
3. **Temporal reasoning**: Depends on Gemini correctly interpreting effective dates from page text — no structured date extraction.
4. **Budget is tight**: 6 calls for complex multi-page queries requires efficient tool selection from Gemini.
5. **LLM non-determinism**: Final answers may vary slightly between runs, though the evidence base is fixed.
6. **No multi-document support**: Current UI loads one active document at a time.
