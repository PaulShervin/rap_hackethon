# RAP Hackathon — One-Page Architecture Memo

## The Core Design Statement

> "The agent decides what should happen. The Harness decides what is allowed to happen."
> "We treat document access as a finite evidence budget rather than unlimited retrieval."

---

## Architecture

**Laya (Tier 1, local Ollama)** → **Gemma 3 (Tier 2, local Ollama)** → **Deterministic Harness** → **4 Document Tools** → **Grounded Answer**

The Harness is the single gateway. Neither Laya nor Gemma can access the PDF directly. **No cloud LLM dependency** — all reasoning is fully local.

---

## Why This Design

Most document QA systems load the entire PDF into context (RAG) or embed it into a vector database. Both approaches give the model unlimited, unauditable access to document content — and neither can enforce a hard retrieval budget.

Our design inverts this: **the agent must earn each piece of evidence** through an explicit, logged, budgeted tool call. The Harness acts as a policy enforcer between the agent's intent and actual document access.

---

## Why Laya

Laya handles simple queries without invoking the more expensive Tier 2 reasoning loop. It detects complexity signals (temporal language, supersession cues, multi-page results) and escalates immediately — **preserving all evidence and budget** accumulated so far. This tiered approach conserves the 6-call budget for cases that need it.

---

## Why Tier 2 Gemma 3 (Ollama)

Gemma 3 handles multi-step reasoning: identifying contradictions, evaluating supersession claims, reasoning about effective dates. Crucially, Gemma only **outputs structured JSON actions** — it never calls a Python function directly. The Harness validates and executes the action, then returns the result to Gemma as updated state. Running via Ollama means zero cloud data exposure — all document content stays local.

---

## Why Deterministic Harness

LLMs are probabilistic. Hard constraints must not depend on model behavior. The Harness enforces:
- **Tool allowlist**: exactly 4 tools, nothing else
- **Argument validation**: page numbers must be positive integers, keywords non-empty
- **6-call budget**: checked before every execution, cannot be overridden by any input
- **Security policy**: document content is evidence, never instructions
- **Evidence cache**: same page/keyword returns cached result without consuming budget
- **Complete trace**: every action logged whether executed, cached, or blocked

---

## Budget Strategy

6 calls per question requires disciplined tool selection:
- `list_documents` / `list_headings`: 1 call each (orientation)
- `search_keyword`: 1 call to locate relevant pages
- `get_page`: 1-3 calls to read evidence

The budget forces the agent to be strategic rather than exhaustive. If the budget is exhausted before sufficient evidence is retrieved, the system returns "Insufficient information." — never a guess.

---

## Security

Document text is **fenced** in `<document_evidence>` XML before entering any LLM prompt. The system prompt explicitly states that document content cannot modify instructions, permissions, or budget. Forbidden tools are rejected at the Harness level regardless of what any document, user, or model output says.

---

## Contradiction & Answerability Strategy

Contradictions are tracked in `AgentState`. A deterministic pre-verifier checks:
1. Were any pages retrieved?
2. Are there unresolved contradictions?
3. Was the budget exhausted with no evidence?

If any check fails → `INSUFFICIENT`. Only then does the LLM verifier confirm answerability. This conservative cascade ensures the system never turns uncertainty into a confident answer.

---

## Known Failure Mode

If a PDF contains extremely dense, multi-page policies that require reading 7+ pages to resolve, the 6-call budget may be insufficient — and the system correctly returns "Insufficient information" rather than hallucinating. This is the intended behavior: the budget is a transparency mechanism, not a limitation to work around.
