import json
import re
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "Mistral-7B-Instruct-v0.2"
MODEL_NAME = str(LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else "mistralai/Mistral-7B-Instruct-v0.2")

app = FastAPI(title="DreamCatcher Model Service")

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
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        if not torch.cuda.is_available():
            model = model.to("cpu")

        def generate_text_impl(prompt: str, max_length: int = 1100, num_return_sequences: int = 1) -> str:
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

        generate_text = generate_text_impl

        if tokenizer is None or model is None or generate_text is None:
            raise RuntimeError("The model setup could not be loaded")

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
