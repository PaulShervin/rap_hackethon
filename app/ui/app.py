"""
Gradio chat UI for the Budgeted Document-Answering Agent.
Local-only: Laya (Tier 1) + Gemma 3 via Ollama (Tier 2). No cloud LLM.
"""
import json
import os

import gradio as gr

from app.main import DocumentAnsweringApp

_app = DocumentAnsweringApp()


def _format_trace(trace: list) -> str:
    if not trace:
        return "No trace entries."
    lines = []
    for entry in trace:
        status = entry.get("status", "")
        action = entry.get("action", "")
        call_num = entry.get("call_number", "?")
        tier = entry.get("tier", "")
        reason = entry.get("reason", "")
        result_summary = entry.get("result_summary", "")
        cached = entry.get("cached", False)
        args = entry.get("arguments", {})
        remaining = entry.get("calls_remaining", "?")

        icon = {
            "SUCCESS": "✅", "BLOCKED": "🚫", "REJECTED": "❌",
            "STOP": "⏹", "CACHE_HIT": "💾", "ERROR": "⚠️",
        }.get(status, "•")
        cached_tag = " **[CACHED — no budget consumed]**" if cached else ""
        args_str = json.dumps(args) if args else "{}"

        lines.append(
            f"{icon} `[{tier}]` Call #{call_num} | **{action}**(`{args_str}`){cached_tag}\n"
            f"   Status: `{status}` | Budget remaining: `{remaining}`\n"
            f"   Reason: {reason}\n"
            f"   Result: {result_summary}\n"
        )
    return "\n".join(lines)


def check_health():
    try:
        h = _app.health_check()
        info = h["laya"]  # laya and tier2 share the same Bedrock provider
        icon = "🟢" if info["available"] else "🔴"
        return (
            f"{icon} **Bedrock (Tier 1 + Tier 2):**\n"
            f"- Model: `{info['model']}`\n"
            f"- Region: `{info['region']}`\n"
            f"- Status: {info['message']}"
        )
    except Exception as exc:
        return f"⚠️ Health check failed: {exc}"


def upload_pdf(pdf_file):
    if pdf_file is None:
        return "No file selected.", ""
    try:
        with open(pdf_file.name, "rb") as f:
            pdf_bytes = f.read()
        filename = os.path.basename(pdf_file.name)
        doc_id = _app.upload_pdf(pdf_bytes, filename)
        docs = _app.get_documents()
        doc_info = next((d for d in docs if d["doc_id"] == doc_id), {})
        status = f"✅ Loaded: **{doc_info.get('title', filename)}** ({doc_info.get('pages', '?')} pages) | `{doc_id}`"
        return status, doc_id
    except Exception as exc:
        return f"❌ Upload failed: {exc}", ""


def ask_question(question: str, doc_id: str):
    if not question.strip():
        yield "", "Please enter a question.", "0 / 6", "—", "—", "—", "—"
        return

    target_doc_id = doc_id.strip() or _app.get_active_doc_id()
    if not target_doc_id:
        yield question, "❌ No document loaded. Please upload a PDF first.", "0 / 6", "—", "—", "—", "—"
        return

    yield question, "⏳ Thinking (Laya → Gemma via Ollama)...", "? / 6", "—", "—", "—", "—"

    try:
        result = _app.ask(question, target_doc_id)
    except Exception as exc:
        yield question, f"❌ Error: {exc}", "0 / 6", "—", "—", "—", "—"
        return

    answer = result.get("answer", "Insufficient information.")
    calls_used = result.get("calls_used", 0)
    calls_max = result.get("calls_max", 6)
    tier = result.get("tier", "?")
    escalated = result.get("escalated", False)
    laya_info = result.get("laya_decision") or {}
    trace = result.get("trace", [])
    verification = result.get("verification") or {}
    sources = result.get("sources", [])
    insufficient = result.get("insufficient", False)

    budget_icon = "🔴" if calls_used >= calls_max else ("🟡" if calls_used >= 4 else "🟢")
    budget_display = f"{budget_icon} **{calls_used} / {calls_max}**"

    tier_display = "🔵 **TIER 1 — Laya**" if tier == "TIER_1" else "🟣 **TIER 2 — Gemma (Ollama)**"
    if escalated:
        tier_display += " *(escalated from Laya)*"

    # Laya panel
    laya_text = "—"
    if laya_info:
        decision = laya_info.get("decision", "?")
        confidence = laya_info.get("confidence", 0)
        reason = laya_info.get("reason", "")
        signals = laya_info.get("complexity_signals", [])
        icon = "✅ HANDLE" if decision == "HANDLE" else "⬆️ ESCALATE"
        laya_text = f"**{icon}** (confidence: {confidence:.0%})\n\n**Reason:** {reason}"
        if signals:
            laya_text += f"\n\n**Complexity signals:** {', '.join(signals)}"

    # Verification panel
    verification_text = "—"
    if verification:
        vstatus = verification.get("status", "?")
        vreason = verification.get("reason", "")
        conflicts = verification.get("unresolved_conflicts", [])
        vicon = "✅" if vstatus == "ANSWERABLE" else "❌"
        verification_text = f"{vicon} **{vstatus}**\n\n{vreason}"
        if conflicts:
            verification_text += "\n\n**Conflicts:**\n" + "\n".join(f"- {c}" for c in conflicts)

    # Answer panel
    if insufficient:
        answer_display = f"⚠️ **INSUFFICIENT INFORMATION**\n\n{answer}"
    elif sources:
        answer_display = f"{answer}\n\n📄 **Sources:** {', '.join(sources)}"
    else:
        answer_display = answer

    trace_display = _format_trace(trace)

    yield "", answer_display, budget_display, tier_display, laya_text, verification_text, trace_display


def build_ui():
    with gr.Blocks(
        title="RAP — Budgeted Document-Answering Agent (Local)",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 📄 RAP — Budgeted Document-Answering Agent
**Local architecture:** Laya (Tier 1, Ollama) → Gemma 3 (Tier 2, Ollama) → Deterministic Harness → 4 Document Tools
**Budget:** Hard limit of **6 document tool calls per question**. Call #7 is always blocked.
**No cloud LLM dependency.** All reasoning runs locally via Ollama.
"""
        )

        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                with gr.Accordion("🔧 Model Health", open=True):
                    health_display = gr.Markdown("*Click to check...*")
                    health_btn = gr.Button("Check Ollama Status", size="sm")

                gr.Markdown("### 1. Upload Document")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Load PDF", variant="primary")
                doc_status = gr.Markdown("No document loaded.")
                doc_id_state = gr.Textbox(visible=False, label="Active doc_id")

                gr.Markdown("### 2. Ask a Question")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the current cancellation policy?",
                    lines=3,
                )
                ask_btn = gr.Button("Ask", variant="primary")

                gr.Markdown("### Budget")
                budget_display = gr.Markdown("0 / 6")
                gr.Markdown("### Active Tier")
                tier_display = gr.Markdown("—")

            # Right column
            with gr.Column(scale=2):
                gr.Markdown("### Answer")
                answer_output = gr.Markdown("*Ask a question to see the answer.*")

                with gr.Accordion("Laya Decision (Tier 1 — Local)", open=True):
                    laya_output = gr.Markdown("—")

                with gr.Accordion("Verification Result", open=True):
                    verification_output = gr.Markdown("—")

                with gr.Accordion("Full Agent Trace", open=False):
                    trace_output = gr.Markdown("—")

        # Wire events
        health_btn.click(fn=check_health, outputs=[health_display])

        upload_btn.click(
            fn=upload_pdf,
            inputs=[pdf_input],
            outputs=[doc_status, doc_id_state],
        )

        _ask_outputs = [
            question_input, answer_output, budget_display,
            tier_display, laya_output, verification_output, trace_output,
        ]

        ask_btn.click(fn=ask_question, inputs=[question_input, doc_id_state], outputs=_ask_outputs)
        question_input.submit(fn=ask_question, inputs=[question_input, doc_id_state], outputs=_ask_outputs)

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )


if __name__ == "__main__":
    main()
