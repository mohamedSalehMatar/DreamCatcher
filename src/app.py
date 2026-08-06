import datetime
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from entry_service import DB_DIR, DATE_FORMAT, build_entry_from_service, list_markdown_entries, save_entry
from journal_rag import ask_question, build_rag_index, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource(show_spinner=False)
def get_rag_assets():
    tokenizer, model = load_model()
    vectordb = build_rag_index()
    return tokenizer, model, vectordb


def render_home_tab() -> None:
    st.title("DreamCatcher")
    st.subheader("Your dream journaling workspace")
    st.write(
        "Use the tabs to record a new dream, preview stored journal entries, and chat with your dream database."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Saved dreams", len(list_markdown_entries()))
    with col2:
        st.metric("Journal tab", "Text-only entry")
    with col3:
        st.metric("Analysis tab", "RAG chat")


def render_journal_tab() -> None:
    st.title("Journal")
    st.info("Enter your dream as plain text only. No extra fields are required.")

    dream_text = st.text_area(
        "Dream description",
        height=240,
        placeholder="Describe the dream here...",
    )

    if st.button("Save dream entry", use_container_width=True):
        if not dream_text.strip():
            st.error("Please describe the dream before saving the entry.")
        else:
            with st.spinner("Generating the dream entry..."):
                try:
                    entry = build_entry_from_service(dream_text, datetime.date.today().strftime(DATE_FORMAT))
                    json_path, md_path = save_entry(entry, datetime.date.today().strftime(DATE_FORMAT))
                    st.success(f"Saved {json_path.name} and {md_path.name} in {DB_DIR}")
                    st.json(entry)
                except Exception as exc:
                    st.error(f"Failed to generate the entry: {exc}")


def render_preview_tab() -> None:
    st.title("Preview")
    st.write("Browse the markdown files saved in the database.")

    md_files = list_markdown_entries()
    if not md_files:
        st.info("No dream entries have been saved yet.")
        return

    selected_name = st.selectbox("Choose a saved dream", [path.name for path in md_files])
    selected_path = DB_DIR / selected_name

    if selected_path.exists():
        content = selected_path.read_text(encoding="utf-8")
        st.markdown(content)


def render_analysis_tab() -> None:
    st.title("Analysis")
    st.write("Ask questions about your stored dreams and the assistant will use the RAG index to respond.")

    tokenizer, model, vectordb = get_rag_assets()

    if vectordb is None:
        st.info("No dream entries are available yet for analysis.")
        return

    query = st.text_input("Ask about your dreams")
    if st.button("Analyze") and query.strip():
        with st.spinner("Searching your dream database..."):
            answer = ask_question(query, vectordb, tokenizer, model)
        st.success("Analysis complete")
        st.write(answer)


def main() -> None:
    st.set_page_config(page_title="DreamCatcher", layout="wide")

    tabs = st.tabs(["Home", "Journal", "Preview", "Analysis"])
    with tabs[0]:
        render_home_tab()
    with tabs[1]:
        render_journal_tab()
    with tabs[2]:
        render_preview_tab()
    with tabs[3]:
        render_analysis_tab()

    st.markdown("---")
    st.caption("DreamCatcher — manage and inspect model-generated dream entries stored in the local src/database folder.")


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        script_path = Path(__file__).resolve()
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script_path),
            "--server.headless",
            "true",
        ]
        raise SystemExit(subprocess.call(command, cwd=str(PROJECT_ROOT)))

    main()
