import streamlit as st
from pathlib import Path
import json
import re
import datetime

# Project database folder
DB_DIR = Path("database")
DB_DIR.mkdir(parents=True, exist_ok=True)

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


st.set_page_config(page_title="DreamCatcher", layout="wide")
st.title("DreamCatcher — Journal Manager")

# Sidebar: list existing entries
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

# Left: display selected entry and JSON (if present)
with col1:
    st.subheader("Selected Entry")
    if selected:
        md_path = DB_DIR / selected
        try:
            md_text = md_path.read_text(encoding='utf-8')
        except Exception as e:
            st.error(f"Failed to read {md_path}: {e}")
            md_text = ""
        st.markdown(md_text)

        # show parsed JSON if a corresponding .json exists
        json_path = md_path.with_suffix('.json')
        if json_path.exists():
            st.subheader("Parsed JSON")
            try:
                obj = json.loads(json_path.read_text(encoding='utf-8'))
                st.json(obj)
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

    else:
        st.info("Select an entry from the sidebar to view it.")

# Right: form to paste model output or raw dream and save
with col2:
    st.subheader("Save Model Output / New Entry")
    today = datetime.date.today().strftime(DATE_FORMAT)
    date_str = st.text_input("Date (DD-MM-YYYY)", value=today)

    st.markdown("Paste the model's JSON block (or raw JSON) or paste the markdown output:")
    input_text = st.text_area("Model output or markdown/JSON", height=300)

    save_name = st.text_input("Filename (optional, without extension)", value="")

    if st.button("Save JSON + Markdown"):
        if not input_text.strip():
            st.error("Please paste the model output or JSON to save.")
        else:
            inner = extract_inner_json(input_text)
            pretty, obj = try_pretty_json(inner)

            # determine filename
            fname_base = save_name.strip() or date_str
            # sanitize filename
            fname_base = re.sub(r"[^0-9A-Za-z._-]", "-", fname_base)

            json_path = DB_DIR / f"{fname_base}.json"
            md_path = DB_DIR / f"{fname_base}.md"

            try:
                # write JSON file (if parsed, write the pretty JSON; else write raw inner)
                with json_path.open('w', encoding='utf-8') as f:
                    f.write(pretty)

                # write markdown: include JSON in a fenced block
                with md_path.open('w', encoding='utf-8') as f:
                    f.write('```json\n')
                    f.write(pretty)
                    f.write('\n```\n')

                st.success(f"Saved {json_path.name} and {md_path.name} in {DB_DIR}")
            except Exception as e:
                st.error(f"Failed to save files: {e}")

    st.markdown("---")
    st.markdown("If you want the app to call the model directly, run this project in an environment with the model and Transformers installed and I can add a button to load the tokenizer/model and generate entries.")


st.markdown("---")
st.caption("DreamCatcher — manage and inspect model-generated dream entries stored in the local `database` folder.")
