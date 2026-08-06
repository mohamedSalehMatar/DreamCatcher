import datetime
from pathlib import Path
import re
import json

import torch

from file_naming import build_output_filename
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "src" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_DIR = PROJECT_ROOT / "Mistral-7B-Instruct-v0.2"
MODEL_NAME = str(LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else "mistralai/Mistral-7B-Instruct-v0.2")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return tokenizer, model


def generate_text(prompt, tokenizer, model, max_length=1100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs][0]


def build_rag_index():
    documents = []
    for md_file in DB_DIR.glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        return None

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding)


def ask_question(query, vectordb, tokenizer, model):
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"You are a helpful assistant. Use the following context to answer the question. Question: {query} Context: {context}"
    return generate_text(prompt, tokenizer, model).strip()


def extract_json_block(text):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return f"```json\n{matches[-1]}\n```" if matches else text


def clean_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1])
    return text


def build_markdown_entry(data, date_value):
    title = data.get("dream-title", "Untitled Dream")
    description = data.get("dream_description", "")
    symbols = data.get("dream_symbols", [])
    vibes = data.get("dream_vibes", [])

    if not isinstance(symbols, list):
        symbols = [symbols]
    if not isinstance(vibes, list):
        vibes = [vibes]

    lines = [
        f"# {title}",
        "",
        f"**Dream Date:** {data.get('dream_date', date_value)}",
        "",
        "## Dream",
        "",
        "## Dream Description",
        description,
        "",
        "## Dream Symbols",
    ]
    lines.extend(f"- {symbol}" for symbol in symbols)
    lines.extend(["", "## Dream Vibes"])
    lines.extend(f"- {vibe}" for vibe in vibes)
    return "\n".join(lines)


def save_entry_from_model_output(output_text, date_value):
    json_text = extract_json_block(output_text)
    clean_text = clean_json_text(json_text)
    try:
        data = json.loads(clean_text)
    except Exception:
        data = {}

    if not isinstance(data, dict):
        data = {}

    json_content = json.dumps(data, indent=2, ensure_ascii=False)
    title = data.get("title") or data.get("dream-title") or "Untitled Dream"
    fname_base = build_output_filename(title, date_value)
    json_path = DB_DIR / f"{fname_base}.json"
    md_path = DB_DIR / f"{fname_base}.md"
    json_path.write_text(json_content, encoding="utf-8")
    md_path.write_text(build_markdown_entry(data, date_value), encoding="utf-8")
    return json_path, md_path


def main():
    tokenizer, model = load_model()
    vectordb = build_rag_index()
    date_value = datetime.date.today().strftime("%d-%m-%Y")
    prompt = "I had a dream where I was running away from a mirror"
    answer = generate_text(prompt, tokenizer, model)
    json_text = extract_json_block(answer)
    save_entry_from_model_output(json_text, date_value)
    if vectordb is not None:
        print(ask_question("What is the most detailed dream that I had?", vectordb, tokenizer, model))


if __name__ == "__main__":
    main()
