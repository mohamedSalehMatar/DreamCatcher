import datetime
import json
import re
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(title="DreamCatcher Model Service")

notebook_path = PROJECT_ROOT / "src" / "main.ipynb"
tokenizer = None
model = None
generate_text = None


class DreamRequest(BaseModel):
    dream_text: str


class DreamResponse(BaseModel):
    dream_title: str
    dream_date: str
    dream_description: str
    dream_symbols: list[str]
    dream_vibes: list[str]


def load_model_once() -> tuple[Any, Any, Any]:
    global tokenizer, model, generate_text
    if tokenizer is None or model is None or generate_text is None:
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace: dict[str, Any] = {
            "__name__": "__main__",
            "Path": Path,
            "datetime": datetime,
            "re": re,
            "torch": torch,
            "json": json,
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        }

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            code = "".join(source) if isinstance(source, list) else str(source)
            if not code.strip():
                continue
            exec(compile(code, str(notebook_path), "exec"), namespace)
            if "def generate_text" in code:
                break

        tokenizer = namespace.get("tokenizer")
        model = namespace.get("model")
        generate_text = namespace.get("generate_text")

        if tokenizer is None or model is None or generate_text is None:
            raise RuntimeError("The notebook model setup could not be loaded")

    return tokenizer, model, generate_text


def extract_inner_json(text: str) -> str:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def parse_model_output(text: str) -> dict[str, Any]:
    cleaned = extract_inner_json(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-dream-entry", response_model=DreamResponse)
def generate_dream_entry(request: DreamRequest) -> DreamResponse:
    tokenizer, model, generate_text = load_model_once()

    prompt = f"""You are an expert dream journaler.
Return ONLY valid JSON with these keys:
- dream-title: a concise title in 3 words or fewer
- dream_date: the date in DD-MM-YYYY format
- dream_description: an exact copy of the user's dream text
- dream_symbols: a list of strings describing symbols in the dream
- dream_vibes: a list of strings describing the main emotional vibes

User dream:
{request.dream_text}
"""

    text = generate_text(prompt)
    parsed = parse_model_output(text)

    if not parsed:
        parsed = {
            "dream-title": "Dream Entry",
            "dream_date": "",
            "dream_description": request.dream_text,
            "dream_symbols": ["dream imagery"],
            "dream_vibes": ["reflective"],
        }

    return DreamResponse(
        dream_title=parsed.get("dream-title", "Dream Entry"),
        dream_date=parsed.get("dream_date", ""),
        dream_description=parsed.get("dream_description", request.dream_text),
        dream_symbols=parsed.get("dream_symbols", ["dream imagery"]),
        dream_vibes=parsed.get("dream_vibes", ["reflective"]),
    )
