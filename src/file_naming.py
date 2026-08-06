import re


def slugify_title(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip(".-")
    if not slug:
        raise ValueError("A non-empty title is required to build the output filename")
    return slug.lower()


def build_output_filename(title: str, date_value: str) -> str:
    title_value = (title or "").strip()
    if not title_value:
        raise ValueError("A non-empty title is required to build the output filename")

    date_text = (date_value or "").strip()
    if not date_text:
        raise ValueError("A non-empty date is required to build the output filename")

    safe_date = re.sub(r"[^0-9A-Za-z._-]+", "-", date_text).strip(".-")
    if not safe_date:
        raise ValueError("A non-empty date is required to build the output filename")

    return f"{safe_date}_{slugify_title(title_value)}"
