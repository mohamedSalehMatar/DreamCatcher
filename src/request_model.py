from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_URL = "http://127.0.0.1:8000/generate-dream-entry"


def generate_entry_via_api(dream_text: str) -> dict:
    try:
        response = requests.post(SERVICE_URL, json={"dream_text": dream_text}, timeout=1800)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "The model service is not running. Start it with: python -m uvicorn src.model_service:app --host 127.0.0.1 --port 8000"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The model service did not respond in time.") from exc
