import json
import re
import datetime
from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException

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
    description="A list of the symbolism in the dream. Mainly 3 symbols and what they mean based on context",
)
dream_vibes_schema = ResponseSchema(
    name="vibes",
    description="A list of the main vibes in the dream. Mainly 3 feelings based on context",
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


def parse_model_output(text: str) -> dict[str, Any]:
    cleaned_text = text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    json_candidates = re.findall(r"\{.*?\}", cleaned_text, re.DOTALL)
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    try:
        return output_parser.parse(cleaned_text)
    except OutputParserException:
        def extract_string(key: str) -> str | None:
            match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', cleaned_text)
            return match.group(1) if match else None

        def extract_array(key: str) -> list[str]:
            match = re.search(rf'"{key}"\s*:\s*(\[[^\]]*\])', cleaned_text, re.DOTALL)
            if not match:
                return []
            try:
                parsed_array = json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
            return [str(item) for item in parsed_array]

        if "title" in cleaned_text and "description" in cleaned_text:
            return {
                "title": extract_string("title"),
                "date": extract_string("date") or datetime.date.today().strftime("%d-%m-%Y"),
                "description": extract_string("description") or cleaned_text,
                "symbols": extract_array("symbols"),
                "vibes": extract_array("vibes"),
            }

        raise


resume_extraction_template = """
You are an expert dream journalist and analyst extracting dream details from the dreamer's dream.

Return ONLY a single JSON object with exactly these keys:
- title
- date
- description
- symbols
- vibes

Now extract from this dream:
"{user_input}"
these keys:
- title
- date
- description
- symbols
- vibes

following:
Formatting instructions:
"{format_instructions}"

and these rules:
Rules:
- Use the exact key names shown above.
- Do not wrap the JSON in markdown fences.
- Do not include commentary, explanations, or extra keys.
- "title" should be a short phrase, not a sentence.
- "date" should be in DD-MM-YYYY format.
- "description" should be the dream text faithfully rendered as a single string.
- "symbols" and "vibes" must be arrays of strings.
"""


def load_model_once() -> tuple[Any, Any, Any]:
    global tokenizer, model, generate_text
    if tokenizer is None or model is None or generate_text is None:
        model_kwargs: dict[str, Any] = {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
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

    try:
        parsed_data = parse_model_output(text)
    except Exception:
        parsed_data = {}

    return DreamResponse(
        dream_title=parsed_data.get("title", []),
        dream_date=parsed_data.get("date", datetime.date.today().strftime("%d-%m-%Y")),
        dream_description=parsed_data.get("description", request.dream_text),
        dream_symbols=parsed_data.get("symbols", []),
        dream_vibes=parsed_data.get("vibes", []),
    )
