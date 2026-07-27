import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
import torch
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project database folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "src" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "Mistral-7B-Instruct-v0.2"
DATE_FORMAT = "%d-%m-%Y"


def extract_inner_json(text: str) -> str:
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    return m.group(1) if m else text


def try_pretty_json(text: str):
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return pretty, obj
    except Exception:
        return text, None


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]", "-", value).strip(".-") or "dream-entry"


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
    title = entry.get("dream-title", "Untitled Dream")
    fname_base = sanitize_filename(suggested_name.strip() or title or date_value)
    json_path = DB_DIR / f"{fname_base}.json"
    md_path = DB_DIR / f"{fname_base}.md"

    pretty = json.dumps(entry, indent=2, ensure_ascii=False)
    markdown_text = format_entry_markdown(entry, date_value)

    with json_path.open("w", encoding="utf-8") as handle:
        handle.write(pretty)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown_text)

    return json_path, md_path


@st.cache_resource(show_spinner=False)
def load_model_components():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_entry_from_dream(dream_text: str, date_value: str, title_hint: str = "") -> dict:
    tokenizer, model, device = load_model_components()

    prompt = f"""You are an expert dream journaler.
Return ONLY valid JSON with these keys:
- dream-title: a concise title in 3 words or fewer
- dream_date: the date in DD-MM-YYYY format
- dream_description: an exact copy of the user's dream text
- dream_symbols: a list of strings describing symbols in the dream
- dream_vibes: a list of strings describing the main emotional vibes

User dream:
{dream_text}

Date to use: {date_value}

If a title is provided, use it as the basis for dream-title.
Title hint: {title_hint}
"""

    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    encoded = encoded.to(device)

    with torch.no_grad():
        output = model.generate(
            encoded,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0][encoded.shape[-1]:], skip_special_tokens=True).strip()
    inner = extract_inner_json(text)
    pretty, parsed = try_pretty_json(inner)
    if isinstance(parsed, dict):
        parsed.setdefault("dream-title", title_hint.strip() or "Untitled Dream")
        parsed.setdefault("dream_date", date_value)
        parsed.setdefault("dream_description", dream_text)
        parsed.setdefault("dream_symbols", [])
        parsed.setdefault("dream_vibes", [])
        return parsed

    raise ValueError(f"The model did not return valid JSON: {text}")


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
        st.subheader("Create a dream entry with the model")
        today = datetime.date.today().strftime(DATE_FORMAT)
        date_str = st.text_input("Date (DD-MM-YYYY)", value=today)
        dream_text = st.text_area(
            "Dream description",
            height=220,
            placeholder="Describe the dream here...",
        )
        title_hint = st.text_input("Optional title", value="")

        if st.button("Generate entry with Mistral", use_container_width=True):
            if not dream_text.strip():
                st.error("Please describe the dream before generating an entry.")
            else:
                with st.spinner("Creating a structured dream entry with the model..."):
                    try:
                        entry = generate_entry_from_dream(dream_text, date_str, title_hint)
                        json_path, md_path = save_entry(entry, date_str, title_hint)
                        st.success(f"Saved {json_path.name} and {md_path.name} in {DB_DIR}")
                        st.json(entry)
                    except Exception as exc:
                        st.error(f"Failed to generate the entry: {exc}")

        st.markdown("---")
        st.subheader("Manual save")
        st.markdown("Paste a JSON block or markdown output if you want to save it directly.")
        input_text = st.text_area("Manual JSON/markdown", height=180)
        manual_name = st.text_input("Manual filename (optional)", value="")

        if st.button("Save manual entry"):
            if not input_text.strip():
                st.error("Please paste the JSON or markdown content to save.")
            else:
                inner = extract_inner_json(input_text)
                pretty, parsed = try_pretty_json(inner)
                if isinstance(parsed, dict):
                    entry = parsed
                else:
                    entry = {"dream-title": manual_name.strip() or "Manual Entry", "dream_description": inner}

                try:
                    json_path, md_path = save_entry(entry, date_str, manual_name)
                    st.success(f"Saved {json_path.name} and {md_path.name} in {DB_DIR}")
                except Exception as exc:
                    st.error(f"Failed to save files: {exc}")

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
