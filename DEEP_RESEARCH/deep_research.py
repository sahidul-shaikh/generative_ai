import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
import asyncio

load_dotenv(override=True)

async def do_research(query: str, clarification_questions: str, clarification_answers: str):
    research_input = f"Original Query: {query}\n\nClarification Questions: {clarification_questions}\n\nClarification Answers: {clarification_answers}"
    async for chunk in ResearchManager().run(research_input):
        yield chunk

async def ask_clarification_questions(query: str):
    return await ResearchManager().generate_clarification_questions(query)

with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
    gr.Markdown("# Deep Research")
    query_textbox = gr.Textbox(label="What ltopic would you like to research?")

    clarify_button = gr.Button("Generate Clarification Questions", variant="primary")
    gr.Markdown("# Clarification Questions")
    clarification_questions = gr.Markdown(label="Clarification Questions")
    clarify_button.click(fn=ask_clarification_questions, inputs=query_textbox, outputs=clarification_questions)
    clarification_answers_textbox = gr.Textbox(label="Answer the clarification questions...")

    run_button = gr.Button("Do Research", variant="primary")
    gr.Markdown("# Report")
    report = gr.Markdown(label="Report")
    run_button.click(fn=do_research, inputs=[query_textbox, clarification_questions, clarification_answers_textbox], outputs=report)

ui.launch(inbrowser=True)