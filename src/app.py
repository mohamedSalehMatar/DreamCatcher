import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from file_naming import build_output_filename
from request_model import generate_entry_via_api

# Project database folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "src" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DATE_FORMAT = "%d-%m-%Y"


def format_entry_markdown(entry: dict, date_value: str) -> str:
    title = entry.get("dream-title", "Untitled Dream")
    description = entry.get("dream_description", "")
    symbols = entry.get("dream_symbols", [])
    vibes = entry.get("dream_vibes", [])

    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(vibes, str):
        vibes = [vibes]

    lines = [f"# {title}", "", f"**Dream Date:** {date_value}", "", "## Dream", "", description, ""]
    lines.extend(["## Dream Symbols", ""])
    lines.extend([f"- {item}" for item in symbols] if symbols else ["- None"])
    lines.extend(["", "## Dream Vibes", ""])
    lines.extend([f"- {item}" for item in vibes] if vibes else ["- None"])
    return "\n".join(lines) + "\n"


def save_entry(entry: dict, date_value: str, suggested_name: str = "") -> tuple[Path, Path]:
    title = (suggested_name.strip() if suggested_name else "") or entry.get("dream-title") or entry.get("title") or entry.get("dream_title")
    if not title:
        raise ValueError("A title is required for the output filename")

    fname_base = build_output_filename(title, date_value)
    json_path = DB_DIR / f"{fname_base}.json"
    md_path = DB_DIR / f"{fname_base}.md"

    pretty = json.dumps(entry, indent=2, ensure_ascii=False)
    markdown_text = format_entry_markdown(entry, date_value)

    with json_path.open("w", encoding="utf-8") as handle:
        handle.write(pretty)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown_text)

    return json_path, md_path


def build_entry_from_service(dream_text: str, date_value: str) -> dict:
    payload = generate_entry_via_api(dream_text)
    # entry = {
    #     "dream-title": payload.get("dream_title", "Dream Entry"),
    #     "dream_date": payload.get("dream_date") or date_value,
    #     "dream_description": payload.get("dream_description", dream_text),
    #     "dream_symbols": payload.get("dream_symbols", ["dream imagery"]),
    #     "dream_vibes": payload.get("dream_vibes", ["reflective"]),
    # }
    return payload


def main() -> None:
    st.set_page_config(page_title="DreamCatcher", layout="wide")
    st.title("DreamCatcher — Journal Manager")

    with st.sidebar:
        st.header("Entries")
        md_files = sorted(DB_DIR.glob("*.md"), reverse=True)
        choices = [p.name for p in md_files]
        selected = st.selectbox("Select entry (markdown)", [""] + choices)
        st.markdown("---")
        st.markdown("## Quick Actions")
        if st.button("Open database folder"):
            st.write(f"Database folder: {DB_DIR.resolve()}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Selected Entry")
        if selected:
            md_path = DB_DIR / selected
            try:
                md_text = md_path.read_text(encoding="utf-8")
            except Exception as exc:
                st.error(f"Failed to read {md_path}: {exc}")
                md_text = ""
            st.markdown(md_text)

            json_path = md_path.with_suffix(".json")
            if json_path.exists():
                st.subheader("Parsed JSON")
                try:
                    obj = json.loads(json_path.read_text(encoding="utf-8"))
                    st.json(obj)
                except Exception as exc:
                    st.error(f"Failed to parse JSON: {exc}")
        else:
            st.info("Select an existing entry from the sidebar to view it.")

    with col2:
        st.subheader("Create a dream entry")
        st.info("Enter a dream below and this app will call the FastAPI model service so the model is loaded once in the server process.")
        dream_text = st.text_area(
            "Dream description",
            height=220,
            placeholder="Describe the dream here...",
        )

        if st.button("Generate entry via model service", use_container_width=True):
            if not dream_text.strip():
                st.error("Please describe the dream before generating an entry.")
            else:
                with st.spinner("Calling the model service..."):
                    try:
                        entry = build_entry_from_service(dream_text, datetime.date.today().strftime(DATE_FORMAT))
                        json_path, md_path = save_entry(entry, datetime.date.today().strftime(DATE_FORMAT))
                        st.success(f"Saved {json_path.name} and {md_path.name} in {DB_DIR}")
                        st.json(entry)
                    except Exception as exc:
                        st.error(f"Failed to generate the entry: {exc}")

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
