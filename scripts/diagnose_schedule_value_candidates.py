#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import (  # noqa: E402
    case_context_text,
    clean_repeated_values,
    constrained_value_from_text,
    tool_context_window,
)
from eval_toolcall_jsonl import tool_schema_by_name  # noqa: E402


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def case_key(row, fallback_idx):
    return row.get("id") or row.get("case_id") or str(fallback_idx)


def parse_target_value(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return str(text).strip().strip('"')


def normalize_for_compare(value):
    return clean_repeated_values(value)


def add_candidate(candidates, value, allow_empty=False):
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
        value = value.strip("`")
        value = value.strip()
        if len(value) > 1 and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1].strip()
        else:
            value = value.rstrip("'\"").strip()
        if not value and not allow_empty:
            return
    if value not in candidates:
        candidates.append(value)


def quoted_string_candidates(text):
    values = []
    for match in re.finditer(r'"([^"\n]{2,100})"', text):
        add_candidate(values, match.group(1))
    for match in re.finditer(r"'([^'\n]{2,100})'", text):
        add_candidate(values, match.group(1))
    return values


def capitalized_model_phrase_candidates(text):
    values = []
    token = r"[A-Z0-9][A-Za-z0-9+#./-]*"
    phrase = rf"\b{token}(?:\s+{token}){{1,7}}\b"
    for match in re.finditer(phrase, text):
        value = match.group(0).strip()
        if not re.search(r"\d", value):
            continue
        if re.fullmatch(r"\d+(?:\s+\d+)*", value):
            continue
        add_candidate(values, value)
    return values


def capitalized_phrase_candidates(text):
    values = []
    token = r"[A-Z][A-Za-z0-9+#./-]*"
    phrase = rf"\b{token}(?:\s+{token}){{1,7}}\b"
    stop_starts = {"As", "For", "Could", "Your", "Thank", "Based", "Given", "Additionally"}
    for match in re.finditer(phrase, text):
        value = match.group(0).strip()
        if value.split()[0] in stop_starts:
            continue
        if re.fullmatch(r"\d+(?:\s+\d+)*", value):
            continue
        add_candidate(values, value)
    return values


def id_like_candidates(text):
    values = []
    patterns = [
        r"\b[A-Z]{2,10}-\d{2,}\b",
        r"\b[A-Z]{2,10}-[A-Z0-9]{2,}\b",
        r"\b[A-Z]{1,6}\d{3,}\b",
        r"\b[A-Za-z]+_\d{2,}\b",
        r"\b[a-z]+(?:-[a-z]+)+-\d{2,}\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            add_candidate(values, match.group(0))
    return values


def currency_code_candidates(text):
    values = []
    for match in re.finditer(r"\b(USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)\b", text):
        add_candidate(values, match.group(1))
    return values


def credit_card_candidates(text):
    values = []
    for match in re.finditer(r"\b(?:\d{4}[ -]){3}\d{4}\b", text):
        add_candidate(values, match.group(0).replace("-", " "))
    return values


def expiry_candidates(text):
    values = []
    for match in re.finditer(r"\b(0[1-9]|1[0-2])/\d{2}\b", text):
        add_candidate(values, match.group(0))
    return values


def code_block_candidates(text):
    values = []
    for match in re.finditer(r"```\s*\n(.*?)\n\s*```", text, flags=re.DOTALL):
        value = "\n".join(line.strip() for line in match.group(1).strip().splitlines())
        add_candidate(values, value)
    return values


def ticker_candidates(text):
    values = []
    for match in re.finditer(r"\b[A-Z]{2,6}\b", text):
        value = match.group(0)
        if value in {"USD", "VIP", "CVV", "AM", "PM"}:
            continue
        add_candidate(values, value)
    return values


def percentage_hundredths_candidates(text):
    values = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*%", text):
        value = float(match.group(1)) * 100
        add_candidate(values, int(value) if value.is_integer() else value)
    return values


def percentage_midpoint_candidates(text):
    values = []
    pattern = r"(-?\d+(?:\.\d+)?)\s*%\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)\s*%"
    for match in re.finditer(pattern, text):
        left = float(match.group(1))
        right = float(match.group(2))
        value = (left + right) / 2.0
        add_candidate(values, int(value) if value.is_integer() else value)
    return values


def equal_weight_candidates(text):
    values = []
    ticker_count = len(ticker_candidates(text))
    if ticker_count == 3:
        add_candidate(values, 0.333)
        add_candidate(values, 0.334)
    return values


def event_id_candidates(text):
    values = []
    venue_match = re.search(r"\bat the ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})", text)
    dates = date_candidates(text)
    if venue_match and dates:
        initials = "".join(part[0] for part in venue_match.group(1).split() if part and part[0].isupper())
        if 2 <= len(initials) <= 6:
            year, month, day = dates[0]
            add_candidate(values, f"{initials}-{day:02d}{month:02d}{year % 100:02d}")
    return values


def iso_date_string_candidates(text):
    values = []
    for year, month, day in date_candidates(text):
        add_candidate(values, f"{year:04d}-{month:02d}-{day:02d}")
    return values


def boolean_candidates(text):
    values = []
    lower = text.lower()
    if re.search(r"\b(true|enable|enabled|activate|activated|turn on|auto[- ]?off|automated|automatic)\b", lower):
        add_candidate(values, True)
    if re.search(r"\b(false|disable|disabled|deactivate|turn off)\b", lower):
        add_candidate(values, False)
    return values


def snake_like_candidates(text):
    values = []
    for match in re.finditer(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text):
        add_candidate(values, match.group(0))
    return values


def symbolic_language_candidates(text):
    values = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:#|\+\+)(?![A-Za-z0-9_])", text):
        add_candidate(values, match.group(0))
    return values


def acronym_candidates(text):
    values = []
    for match in re.finditer(r"\b[A-Z][A-Za-z]{1,5}\b", text):
        value = match.group(0)
        if value.lower() in MONTHS:
            continue
        if value in {"AM", "PM", "USD"}:
            continue
        add_candidate(values, value)
    return values


def phrase_candidates_for_key(text, key):
    values = []
    lower_key = str(key or "").lower()
    if lower_key in {"room", "location", "area", "door"} or any(
        token in lower_key for token in ["room", "location", "area", "door"]
    ):
        add_candidate(values, "", allow_empty=True)
        descriptors = [
            "living room",
            "front door",
            "back door",
            "side door",
            "garage door",
            "kitchen",
            "bedroom",
            "bathroom",
            "dining room",
            "front garden",
            "garden",
            "office",
            "garage",
        ]
        lowered = text.lower()
        for phrase in descriptors:
            if phrase in lowered:
                add_candidate(values, phrase)
    if "command" in lower_key:
        for match in re.finditer(
            r"\b(Activate|Set|Turn|Start|Stop|Open|Close|Lock|Unlock|Record|Retrieve)\b[^.?!\n,;:]{2,120}",
            text,
        ):
            value = match.group(0).strip()
            value = re.sub(r"\s+(?:please|using|with|for|and)$", "", value, flags=re.IGNORECASE).strip()
            value = value.strip("'\"")
            add_candidate(values, value)
    if "condition" in lower_key:
        lowered = text.lower()
        if re.search(r"\btemperature\b.{0,80}\babove\b", lowered):
            add_candidate(values, "temperature_above")
        if re.search(r"\btemperature\b.{0,80}\bbelow\b", lowered):
            add_candidate(values, "temperature_below")
        for word in [
            "sunny",
            "clear",
            "cloudy",
            "rainy",
            "raining",
            "overcast",
            "snowy",
            "windy",
        ]:
            if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
                add_candidate(values, word)
    if "demographic" in lower_key:
        for phrase in [
            "young adults",
            "professionals",
            "students",
            "parents",
            "seniors",
            "teenagers",
            "families",
        ]:
            if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
                add_candidate(values, phrase)
    if "priority" in lower_key:
        for phrase in ["high", "medium", "low"]:
            if re.search(rf"\b{phrase}\b", text, flags=re.IGNORECASE):
                add_candidate(values, phrase)
    if "expense_type" in lower_key or lower_key == "type":
        for phrase in ["materials", "material", "labor", "overhead", "VIP", "General", "Balcony"]:
            if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
                add_candidate(values, phrase)
    if "ticket_type" in lower_key:
        for phrase in ["VIP", "General", "Balcony"]:
            if re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE):
                add_candidate(values, phrase)
    if "device_type" in lower_key or lower_key == "type":
        device_phrases = {
            "smart lock": "smart_lock",
            "smart light": "smart_light",
            "thermostat": "thermostat",
        }
        lowered = text.lower()
        for phrase, rendered in device_phrases.items():
            if phrase in lowered:
                add_candidate(values, rendered)
    if "action_command" in lower_key or lower_key == "command":
        if re.search(r"\bset\b", text, flags=re.IGNORECASE) and re.search(r"\btemperature\b", text, flags=re.IGNORECASE):
            add_candidate(values, "set_temperature")
    if "time_range" in lower_key:
        if re.search(r"\blast\s+24\s+hours?\b", text, flags=re.IGNORECASE):
            add_candidate(values, "last_24_hours")
    if "session_id" in lower_key and "create_trivia_game_session" in text:
        add_candidate(values, "use_id_from_create_trivia_game_session")
    if lower_key == "name":
        scenario_map = {
            "optimistic": "Optimistic",
            "pessimistic": "Pessimistic",
            "neutral": "Neutral",
        }
        lowered = text.lower()
        for word, rendered in scenario_map.items():
            if re.search(rf"\b{word}\b", lowered):
                add_candidate(values, rendered)
    return values


def inline_field_value_candidates(text, key):
    values = []
    raw_key = str(key or "")
    if not raw_key:
        return values
    key_patterns = {
        raw_key,
        raw_key.replace("_", " "),
        raw_key.replace("_", "-"),
    }
    for key_text in sorted(key_patterns, key=len, reverse=True):
        if not key_text:
            continue
        pattern = rf"(?:`{re.escape(raw_key)}`|{re.escape(key_text)})\s*[:=]\s*(?:\"([^\"]*)\"|'([^']*)'|([A-Za-z0-9][A-Za-z0-9 _./+#-]{{0,100}}))"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = next((group for group in match.groups() if group is not None), "")
            value = re.split(r"\s*(?:\n|;|, and | and then | then )", value.strip(), maxsplit=1)[0].strip()
            add_candidate(values, value, allow_empty=True)
    return values


def markdown_table_values(text):
    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        for cell in cells:
            if cell and not re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")):
                add_candidate(values, cell)
    return values


def normalize_field_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def markdown_tables(text):
    tables = []
    active = []
    for line in text.splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            active.append(stripped)
            continue
        if active:
            tables.append(active)
            active = []
    parsed = []
    for table in tables:
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table]
        rows = [row for row in rows if row]
        if len(rows) < 2:
            continue
        header = rows[0]
        body = []
        for row in rows[1:]:
            if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row):
                continue
            if len(row) == len(header):
                body.append(row)
        if body:
            parsed.append({"header": header, "rows": body})
    return parsed


def collection_from_path(path):
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\.([A-Za-z_][A-Za-z0-9_]*)$", str(path or ""))
    if not match:
        return None, None, None
    return match.group(1), int(match.group(2)), match.group(3)


def markdown_table_field_candidates(text, key, path):
    collection, row_idx, path_key = collection_from_path(path)
    if collection is None:
        return []
    wanted_key = normalize_field_name(path_key or key)
    collection_hints = {
        "expense_data": {"date", "category", "amount", "description"},
        "invoice_data": {"invoiceid", "clientid", "amount", "duedate"},
        "payment_data": {"paymentid", "invoiceid", "amount", "datereceived"},
    }
    expected_headers = collection_hints.get(collection)
    values = []
    for table in markdown_tables(text):
        header_norm = [normalize_field_name(item) for item in table["header"]]
        if expected_headers and not expected_headers.issubset(set(header_norm)):
            continue
        if wanted_key not in header_norm:
            continue
        if row_idx >= len(table["rows"]):
            continue
        cell = table["rows"][row_idx][header_norm.index(wanted_key)]
        add_candidate(values, cell)
    return values


def numeric_candidates(text, expected_type):
    values = []
    number_patterns = [
        r"(?<![A-Za-z0-9_])-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        r"\b-?\d+(?:\.\d+)?\b",
    ]
    seen_spans = set()
    for pattern in number_patterns:
        for match in re.finditer(pattern, text):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            raw = match.group(0).replace("$", "").replace(",", "")
            if expected_type == "integer":
                try:
                    add_candidate(values, int(float(raw)))
                except Exception:
                    pass
            elif expected_type == "number":
                try:
                    number = float(raw)
                    add_candidate(values, int(number) if number.is_integer() else number)
                except Exception:
                    pass
    return values


def schema_coerced_candidates(value, expected_type):
    values = []
    add_candidate(values, value, allow_empty=True)
    if isinstance(value, str) and expected_type in {"integer", "number"}:
        raw = value.strip().replace("$", "").replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
            try:
                number = float(raw)
                if expected_type == "integer":
                    add_candidate(values, int(number))
                else:
                    add_candidate(values, int(number) if number.is_integer() else number)
            except Exception:
                pass
    return values


def add_schema_candidate(candidates, value, expected_type, allow_empty=False):
    for candidate in schema_coerced_candidates(value, expected_type):
        add_candidate(candidates, candidate, allow_empty=allow_empty)


def legacy_numeric_candidates(text, expected_type):
    values = []
    for match in re.finditer(r"\b-?\d+(?:\.\d+)?\b", text):
        raw = match.group(0)
        if expected_type == "integer":
            try:
                add_candidate(values, int(float(raw)))
            except Exception:
                pass
        elif expected_type == "number":
            try:
                number = float(raw)
                add_candidate(values, int(number) if number.is_integer() else number)
            except Exception:
                pass
    return values


def time_candidates(text):
    times = []
    for match in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text):
        add_candidate(times, (int(match.group(1)), int(match.group(2))))
    for match in re.finditer(r"\b(1[0-2]|0?[1-9])\s*(AM|PM)\b", text, flags=re.IGNORECASE):
        hour = int(match.group(1))
        suffix = match.group(2).lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        add_candidate(times, (hour, 0))
    return times


def date_candidates(text):
    dates = []
    month_names = "|".join(MONTHS)
    for match in re.finditer(rf"\b({month_names})\s+(\d{{1,2}}),\s*(\d{{4}})\b", text, flags=re.IGNORECASE):
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        add_candidate(dates, (year, month, day))
    years = [int(match.group(1)) for match in re.finditer(r"\b(20\d{2}|19\d{2})\b", text)]
    fallback_year = years[-1] if years else 2023
    for match in re.finditer(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else fallback_year
        add_candidate(dates, (year, month, day))
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        add_candidate(dates, (int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return dates


def datetime_candidates(text):
    values = []
    dates = date_candidates(text)
    times = time_candidates(text)
    for year, month, day in dates:
        for hour, minute in times:
            add_candidate(values, f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z")
    return values


def clock_string_candidates(text):
    values = []
    for hour, minute in time_candidates(text):
        add_candidate(values, f"{hour:02d}:{minute:02d}")
    return values


def paired_datetime_candidate(text, key):
    lower_key = str(key or "").lower()
    if "time" not in lower_key:
        return None
    if not any(token in lower_key for token in ["start", "begin", "end", "stop"]):
        return None
    dates = date_candidates(text)
    times = time_candidates(text)
    if not dates or len(times) < 2:
        return None
    year, month, day = dates[-1]
    if any(token in lower_key for token in ["start", "begin"]):
        hour, minute = times[0]
    else:
        hour, minute = times[-1]
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def schema_for_path(root_schema, path, fallback_key=None):
    if not isinstance(root_schema, dict):
        return {}
    if not path:
        return (root_schema.get("properties") or {}).get(fallback_key, {}) if fallback_key else {}
    schema = root_schema
    for part in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", str(path)):
        key, index = part
        if key:
            if schema.get("type") == "array":
                schema = schema.get("items") or {}
            schema = (schema.get("properties") or {}).get(key, {})
        elif index:
            schema = schema.get("items") or {}
        if not isinstance(schema, dict) or not schema:
            return (root_schema.get("properties") or {}).get(fallback_key, {}) if fallback_key else {}
    return schema if isinstance(schema, dict) else {}


def evidence_candidate_set(context, key, schema, single_candidate, path=None):
    candidates = []
    expected = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), expected[0] if expected else None)
    add_schema_candidate(candidates, single_candidate, expected)
    lower = str(key or "").lower()
    description = str(schema.get("description") or "").lower() if isinstance(schema, dict) else ""
    text_hint = any(
        token in lower or token in description
        for token in [
            "name",
            "model",
            "product",
            "vendor",
            "title",
            "service",
            "stream",
            "workspace",
            "group",
            "target",
            "door",
            "language",
        ]
    )
    if isinstance(schema, dict) and schema.get("enum"):
        for value in schema["enum"]:
            add_candidate(candidates, value)
    table_values = markdown_table_field_candidates(context, key, path)
    if table_values:
        for value in table_values:
            add_schema_candidate(candidates, value, expected)
        if expected in {"string", None}:
            return candidates
    if expected == "boolean":
        for value in boolean_candidates(context):
            add_candidate(candidates, value)
    numeric_expected = expected
    if expected is None and any(token in lower for token in ["amount", "temperature", "duration", "days"]):
        numeric_expected = "number"
    if numeric_expected in {"integer", "number"}:
        for value in numeric_candidates(context, numeric_expected):
            add_candidate(candidates, value)
        if "growth" in lower or "rate" in lower or "growth" in description:
            for value in percentage_midpoint_candidates(context):
                add_candidate(candidates, value)
        if "weight" in lower or "portfolio" in description:
            for value in equal_weight_candidates(context):
                add_candidate(candidates, value)
        if "tax" in lower or "hundredths" in description:
            for value in percentage_hundredths_candidates(context):
                add_candidate(candidates, value)
    if expected == "string" or expected is None:
        if "date" in lower:
            for value in iso_date_string_candidates(context):
                add_candidate(candidates, value)
            if "expiry" in lower or "expiration" in description:
                for value in expiry_candidates(context):
                    add_candidate(candidates, value)
        if "time" in lower or "iso 8601" in str(schema.get("description") or "").lower():
            for value in clock_string_candidates(context):
                add_candidate(candidates, value)
            for value in datetime_candidates(context):
                add_candidate(candidates, value)
        if (
            "id" in lower
            or "identifier" in description
            or "code" in lower
            or "code" in description
            or "device" in lower
            or "device" in description
        ):
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
            for value in id_like_candidates(context):
                add_candidate(candidates, value)
            for value in snake_like_candidates(context):
                add_candidate(candidates, value)
            if "event" in lower:
                for value in event_id_candidates(context):
                    add_candidate(candidates, value)
        if "currency" in lower or "currency" in description:
            for value in currency_code_candidates(context):
                add_candidate(candidates, value)
        if "ticker" in lower or "ticker" in description:
            for value in ticker_candidates(context):
                add_candidate(candidates, value)
        if "card" in lower and "number" in lower:
            for value in credit_card_candidates(context):
                add_candidate(candidates, value)
        if lower == "cvv" or "cvv" in description:
            for match in re.finditer(r"\bCVV(?:\s+of)?\s+(\d{3,4})\b", context, flags=re.IGNORECASE):
                add_candidate(candidates, match.group(1))
        if "log" in lower or "log" in description:
            for value in code_block_candidates(context):
                add_candidate(candidates, value)
        if "mode" in lower or "mode" in description:
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
            for value in snake_like_candidates(context):
                add_candidate(candidates, value)
        if "language" in lower or "language" in description:
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
            for value in symbolic_language_candidates(context):
                add_candidate(candidates, value)
        if "risk" in lower or "model" in lower or "risk" in description:
            for value in acronym_candidates(context):
                add_candidate(candidates, value)
        if text_hint:
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
            for value in capitalized_phrase_candidates(context):
                add_candidate(candidates, value)
        if "model" in lower or "model" in description or "product" in lower or "product" in description:
            for value in capitalized_model_phrase_candidates(context):
                add_candidate(candidates, value)
            for value in capitalized_phrase_candidates(context):
                add_candidate(candidates, value)
        if lower in {"location", "room", "area", "door"} or "location" in description or "room" in description:
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
        if lower in {"category", "description", "target"} or any(
            token in description for token in ["category", "description", "target"]
        ):
            for value in quoted_string_candidates(context):
                add_candidate(candidates, value)
            for value in markdown_table_values(context):
                add_candidate(candidates, value)
            if lower == "target" or "target" in description:
                for value in capitalized_phrase_candidates(context):
                    add_candidate(candidates, value)
        for value in inline_field_value_candidates(context, key):
            add_schema_candidate(candidates, value, expected, allow_empty=True)
        for value in phrase_candidates_for_key(context, key):
            add_schema_candidate(candidates, value, expected, allow_empty=True)
        for value in quoted_string_candidates(context):
            if str(value).lower() in {"1080p", "720p", "480p"}:
                add_candidate(candidates, value)
    return candidates


def tool_names_by_index(schedule):
    names = {}
    for item in schedule:
        if item.get("kind") != "tool_name":
            continue
        idx = item.get("tool_call_index")
        target_text = item.get("target_text")
        if idx is None or target_text is None:
            continue
        names[int(idx)] = parse_target_value(target_text)
    return names


def argument_items(schedule):
    seen = set()
    out = []
    for item in schedule:
        if item.get("kind") != "argument_value":
            continue
        key = (
            item.get("source_token_block_idx"),
            item.get("tool_call_index"),
            item.get("json_key"),
            item.get("target_text"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def diagnose_record(schedule_row, case):
    schedule = schedule_row.get("schedule") or []
    schemas = tool_schema_by_name(case.get("tools") or [])
    context = case_context_text(case)
    tool_names = tool_names_by_index(schedule)
    rows = []
    for item in argument_items(schedule):
        tool_idx = item.get("tool_call_index")
        tool_name = tool_names.get(int(tool_idx)) if tool_idx is not None else None
        arg_key = item.get("json_key")
        target = parse_target_value(item.get("target_text"))
        schema = {}
        if tool_name in schemas and arg_key:
            root_schema = schemas[tool_name]
            schema = schema_for_path(
                root_schema,
                item.get("json_path") or item.get("argument_path"),
                fallback_key=arg_key,
            )
        scoped = tool_context_window(context, tool_name) if tool_name else ""
        candidate_scoped = constrained_value_from_text(scoped, arg_key, schema) if scoped and arg_key else None
        candidate_full = constrained_value_from_text(context, arg_key, schema) if arg_key else None
        if arg_key and candidate_scoped is None:
            candidate_scoped = paired_datetime_candidate(scoped, arg_key) if scoped else None
        if arg_key and candidate_full is None:
            candidate_full = paired_datetime_candidate(context, arg_key)
        candidate = candidate_scoped if candidate_scoped is not None else candidate_full
        candidate_set = evidence_candidate_set(
            context,
            arg_key,
            schema,
            candidate,
            path=item.get("json_path") or item.get("argument_path"),
        )
        if arg_key and str(arg_key).lower().endswith("id") and tool_idx is not None:
            current_idx = int(tool_idx)
            for prev_idx in range(current_idx):
                previous_name = tool_names.get(prev_idx)
                if previous_name:
                    add_candidate(candidate_set, f"use_id_from_{previous_name}")
        exact = normalize_for_compare(candidate) == normalize_for_compare(target)
        target_in_candidates = normalize_for_compare(target) in [
            normalize_for_compare(item) for item in candidate_set
        ]
        rows.append(
            {
                "tool_call_index": tool_idx,
                "tool_name": tool_name,
                "json_key": arg_key,
                "target": target,
                "candidate": candidate,
                "candidate_set": candidate_set,
                "candidate_source": "scoped" if candidate_scoped is not None else "full" if candidate_full is not None else "none",
                "exact": exact,
                "target_in_candidates": target_in_candidates,
                "schema_type": schema.get("type") if isinstance(schema, dict) else None,
                "schema_description": schema.get("description") if isinstance(schema, dict) else None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    args = parser.parse_args()

    cases = {case_key(row, idx): row for idx, row in enumerate(load_jsonl(args.cases_jsonl))}
    records = []
    totals = Counter()
    for idx, row in enumerate(load_jsonl(args.schedule_jsonl)):
        key = case_key(row, idx)
        case = cases.get(key)
        if not case:
            totals["missing_case_records"] += 1
            continue
        diagnostics = diagnose_record(row, case)
        totals["records"] += 1
        totals["argument_values"] += len(diagnostics)
        totals["exact_candidates"] += sum(int(item["exact"]) for item in diagnostics)
        totals["target_in_candidate_sets"] += sum(int(item["target_in_candidates"]) for item in diagnostics)
        for item in diagnostics:
            totals[f"candidate_source:{item['candidate_source']}"] += 1
            totals[f"key:{item['json_key']}"] += 1
        records.append(
            {
                "id": key,
                "source": row.get("source"),
                "tool_call_count": row.get("tool_call_count"),
                "diagnostics": diagnostics,
            }
        )

    summary = {
        "schedule_jsonl": str(args.schedule_jsonl),
        "cases_jsonl": str(args.cases_jsonl),
        "out_jsonl": str(args.out_jsonl) if args.out_jsonl else None,
        "totals": dict(totals),
    }
    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        args.out_jsonl.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
