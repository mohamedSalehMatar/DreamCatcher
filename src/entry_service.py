import json
from pathlib import Path

from file_naming import build_output_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "src" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DATE_FORMAT = "%d-%m-%Y"


def normalize_entry_for_storage(entry: dict, date_value: str) -> dict:
    normalized = dict(entry)
    normalized["dream_date"] = date_value
    if "date" in normalized:
        normalized["date"] = date_value
    return normalized


def format_entry_markdown(entry: dict, date_value: str) -> str:
    title = entry.get("dream_title", [])
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


def list_markdown_entries(database_dir: Path | None = None) -> list[Path]:
    base_dir = database_dir or DB_DIR
    return sorted(base_dir.glob("*.md"), key=lambda path: path.name.lower())


def save_entry(entry: dict, date_value: str, suggested_name: str = "") -> tuple[Path, Path]:
    normalized_entry = normalize_entry_for_storage(entry, date_value)
    title = (
        (suggested_name.strip() if suggested_name else "")
        or normalized_entry.get("dream-title")
        or normalized_entry.get("title")
        or normalized_entry.get("dream_title")
    )
    if not title:
        raise ValueError("A title is required for the output filename")

    fname_base = build_output_filename(title, date_value)
    json_path = DB_DIR / f"{fname_base}.json"
    md_path = DB_DIR / f"{fname_base}.md"

    pretty = json.dumps(normalized_entry, indent=2, ensure_ascii=False)
    markdown_text = format_entry_markdown(normalized_entry, date_value)

    with json_path.open("w", encoding="utf-8") as handle:
        handle.write(pretty)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown_text)

    return json_path, md_path


def build_entry_from_service(dream_text: str, date_value: str) -> dict:
    from request_model import generate_entry_via_api

    payload = generate_entry_via_api(dream_text)
    return payload
