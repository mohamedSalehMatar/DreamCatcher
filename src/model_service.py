import json
import re
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

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


dream_title_schema = ResponseSchema(
    name="title",
    description="A concise title in 3 words or fewer that best describe the user's dream",
)
dream_date_schema = ResponseSchema(
    name="date",
    description="The date of today in DD-MM-YYYY format",
)
dream_description_schema = ResponseSchema(
    name="description",
    description="An exact copy of the user's dream text input",
)
dream_symbols_schema = ResponseSchema(
    name="symbols",
    description="A list of the symbolism in the dream. Mainly 5 symbols and what they mean based on context",
)
dream_vibes_schema = ResponseSchema(
    name="vibes",
    description="A list of the main vibes in the dream. Mainly 5 feelings based on context",
)

response_schemas = [
    dream_title_schema,
    dream_date_schema,
    dream_description_schema,
    dream_symbols_schema,
    dream_vibes_schema,
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

resume_extraction_template = """
You are an expert dream journalist and analyst extracting dream details from the dreamer's dream.

Return ONLY a single JSON object with exactly these keys:
- title
- date
- description
- symbols
- vibes

Rules:
- Use the exact key names shown above.
- Do not wrap the JSON in markdown fences.
- Do not include commentary, explanations, or extra keys.
- "title" should be a short phrase, not a sentence.
- "date" should be in DD-MM-YYYY format.
- "description" should be the dream text faithfully rendered as a single string.
- "symbols" and "vibes" must be arrays of strings.

Example JSON:
{"title": "Glass Symphony", "date": "05-08-2026", "description": "I was flying through a city of glass.", "symbols": ["glass", "flight", "music"], "vibes": ["wonder", "calm"]}

Now extract from this dream:
"{user_input}"
"""


def load_model_once() -> tuple[Any, Any, Any]:
    global tokenizer, model, generate_text
    if tokenizer is None or model is None or generate_text is None:
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

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


def parse_key_value_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}

    normalized: dict[str, Any] = {}
    field_patterns = {
        "title": [r"(?im)^\s*(?:dream\s+)?title\s*[:\-]?\s*(.+)$", r"(?im)^\s*#\s*(.+)$"],
        "date": [r"(?im)^\s*(?:dream\s+)?date\s*[:\-]?\s*(.+)$"],
        "description": [r"(?im)^\s*(?:dream\s+)?description\s*[:\-]?\s*(.+)$"],
        "symbols": [r"(?im)^\s*(?:dream\s+)?symbols\s*[:\-]?\s*(.+)$"],
        "vibes": [r"(?im)^\s*(?:dream\s+)?vibes\s*[:\-]?\s*(.+)$"],
    }

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        for field_name, patterns in field_patterns.items():
            if field_name in normalized:
                continue
            for pattern in patterns:
                match = re.match(pattern, line)
                if not match:
                    continue
                value = match.group(1).strip()
                if field_name in {"symbols", "vibes"}:
                    normalized[field_name] = [part.strip().strip("\"'[]()") for part in re.split(r"\n|,\s*|\s*[-*•]\s*", value) if part.strip()]
                else:
                    normalized[field_name] = value
                break

    if normalized:
        return normalized

    for field_name, pattern in {
        "title": r"(?is)(?:dream\s+)?title\s*[:\-]?\s*(.+?)(?=\n\s*(?:dream\s+)?(?:date|description|symbols|vibes)|$)",
        "date": r"(?is)(?:dream\s+)?date\s*[:\-]?\s*(.+?)(?=\n\s*(?:dream\s+)?(?:title|description|symbols|vibes)|$)",
        "description": r"(?is)(?:dream\s+)?description\s*[:\-]?\s*(.+?)(?=\n\s*(?:dream\s+)?(?:title|date|symbols|vibes)|$)",
        "symbols": r"(?is)(?:dream\s+)?symbols\s*[:\-]?\s*(.+?)(?=\n\s*(?:dream\s+)?(?:title|date|description|vibes)|$)",
        "vibes": r"(?is)(?:dream\s+)?vibes\s*[:\-]?\s*(.+?)(?=\n\s*(?:dream\s+)?(?:title|date|description|symbols)|$)",
    }.items():
        match = re.search(pattern, cleaned)
        if match:
            value = match.group(1).strip()
            if field_name in {"symbols", "vibes"}:
                normalized[field_name] = [part.strip().strip("\"'[]()") for part in re.split(r"\n|,\s*|\s*[-*•]\s*", value) if part.strip()]
            else:
                normalized[field_name] = value

    return normalized


def parse_model_output(text: str) -> dict[str, Any]:
    cleaned = extract_inner_json(text)
    parsed: Any = {}

    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            parsed = output_parser.parse(cleaned)
        except Exception:
            parsed = parse_key_value_response(cleaned)

    if not isinstance(parsed, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in parsed.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in {"symbols", "vibes"} and isinstance(value, str):
            items = [part.strip().strip("\"'[]()") for part in re.split(r"\n|,\s*|\s*[-*•]\s*", value) if part.strip()]
            normalized[normalized_key] = items
        else:
            normalized[normalized_key] = value
    return normalized


def coerce_list(value: Any) -> list[str]:
    if value is None:
        raise ValueError("The model response did not return a valid list for symbols or vibes.")
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            raise ValueError("The model response did not return a valid list for symbols or vibes.")
        return items
    if isinstance(value, str):
        parts = [part.strip().strip("\"'[]()") for part in re.split(r"\n|,\s*|\s*[-*•]\s*", value) if part.strip()]
        if not parts:
            raise ValueError("The model response did not return a valid list for symbols or vibes.")
        return parts
    raise ValueError("The model response did not return a valid list for symbols or vibes.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-dream-entry", response_model=DreamResponse)
def generate_dream_entry(request: DreamRequest) -> DreamResponse:
    tokenizer, model, generate_text = load_model_once()

    prompt = PromptTemplate(
        template=resume_extraction_template,
        input_variables=["user_input", "format_instructions"],
    ).format(user_input=request.dream_text, format_instructions=format_instructions)

    text = generate_text(prompt)
    parsed = parse_model_output(text)

    dream_title = parsed.get("title") or parsed.get("dream_title")
    dream_date = parsed.get("date") or parsed.get("dream_date")
    dream_description = parsed.get("description") or parsed.get("dream_description")
    dream_symbols_raw = parsed.get("symbols") or parsed.get("dream_symbols")
    dream_vibes_raw = parsed.get("vibes") or parsed.get("dream_vibes")

    missing_fields = [
        field_name
        for field_name, value in {
            "title": dream_title,
            "date": dream_date,
            "description": dream_description,
            "symbols": dream_symbols_raw,
            "vibes": dream_vibes_raw,
        }.items()
        if value in (None, "", [], {})
    ]
    if missing_fields:
        raise ValueError(f"The model response is missing required dream fields: {', '.join(missing_fields)}")

    return DreamResponse(
        dream_title=str(dream_title),
        dream_date=str(dream_date),
        dream_description=str(dream_description),
        dream_symbols=coerce_list(dream_symbols_raw),
        dream_vibes=coerce_list(dream_vibes_raw),
    )
