#!/usr/bin/env python3
"""Deterministic scorers for the per-capability conversion-tax battery (#28).

Three capability classes, all scored strictly + deterministically on the decoded
completion text (skip_special_tokens=True, no post-hoc normalization beyond what
is documented here):

  A GSM8K free-CoT : last `#### <number>` == gold (strict, matches L1 harness).
  B CODE (MBPP)    : first ```python fenced block exec'd against the problem's
                     test_imports + test_list asserts in a subprocess (5s wall).
  C INSTRUCTION    : per-item verifiable constraint dispatched on `check.type`.

Every function here is pure / deterministic given the completion string, so the
same completion always scores identically across the 3 systems.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- A: GSM8K ----
_GSM_RE = re.compile(r"####\s*(-?[0-9][0-9,]*)")


def strict_gsm8k_answer(text: str):
    m = _GSM_RE.findall(text or "")
    return m[-1].replace(",", "") if m else None


def score_gsm8k(completion: str, gold_answer: str) -> bool:
    pred = strict_gsm8k_answer(completion)
    gold = strict_gsm8k_answer(gold_answer)
    return pred is not None and gold is not None and pred == gold


# ------------------------------------------------------------------ B: CODE ---
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(completion: str) -> str:
    """First fenced code block; fall back to the raw completion if unfenced."""
    m = _FENCE_RE.search(completion or "")
    if m:
        return m.group(1)
    # tolerate a fence opened but not closed (truncation)
    m2 = re.search(r"```(?:python|py)?\s*\n(.*)$", completion or "", re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1)
    return completion or ""


def score_code(completion: str, test_imports, test_list, timeout: float = 5.0) -> bool:
    code = extract_code(completion)
    parts = []
    parts.extend(test_imports or [])
    parts.append(code)
    parts.extend(test_list or [])
    prog = "\n".join(parts) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(prog)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        ok = (r.returncode == 0)
    except subprocess.TimeoutExpired:
        ok = False
    except Exception:
        ok = False
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass
    return ok


# ----------------------------------------------------------- C: INSTRUCTION ---
def _words(t):
    return t.split()


def _sentence_count(t):
    return len([s for s in re.split(r"[.!?]+", t) if s.strip()])


def _nonempty_lines(t):
    return [ln for ln in t.splitlines() if ln.strip()]


def _extract_json_obj(t):
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


def score_instruction(completion: str, check: dict) -> bool:
    t = (completion or "").strip()
    typ = check["type"]
    low = t.lower()
    if typ == "exact_match":
        return t == check["target"]
    if typ == "word_count_eq":
        return len(_words(t)) == check["n"]
    if typ == "word_count_lt":
        return 0 < len(_words(t)) < check["n"]
    if typ == "word_count_ge":
        return len(_words(t)) >= check["n"]
    if typ == "sentence_count_eq":
        return _sentence_count(t) == check["n"]
    if typ == "all_uppercase":
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if typ == "keyword_count_eq":
        n = len(re.findall(r"\b" + re.escape(check["word"]) + r"\b", low))
        return n == check["n"]
    if typ == "keyword_count_ge":
        n = len(re.findall(r"\b" + re.escape(check["word"]) + r"\b", low))
        return n >= check["n"]
    if typ == "keyword_present":
        return check["word"].lower() in low
    if typ == "contains_all":
        return all(w.lower() in low for w in check["words"])
    if typ == "no_letter":
        return check["letter"].lower() not in low
    if typ == "no_word":
        return re.search(r"\b" + re.escape(check["word"]) + r"\b", low) is None
    if typ == "no_comma":
        return "," not in t
    if typ == "ends_with":
        return t.rstrip().endswith(check["phrase"])
    if typ == "ends_with_char":
        return t.rstrip().endswith(check["char"])
    if typ == "starts_with":
        return t.startswith(check["phrase"])
    if typ == "start_and_end":
        return t.startswith(check["start"]) and t.rstrip().endswith(check["end"])
    if typ == "json_keys":
        obj = _extract_json_obj(t)
        return isinstance(obj, dict) and set(obj.keys()) == set(check["keys"])
    if typ == "line_prefix":
        lines = _nonempty_lines(t)
        return len(lines) == check["n"] and all(ln.startswith(check["prefix"]) for ln in lines)
    if typ == "contains_digit":
        return any(c.isdigit() for c in t)
    raise ValueError(f"unknown check type {typ!r}")


def score_row(clazz: str, completion: str, meta: dict) -> bool:
    if clazz == "A":
        return score_gsm8k(completion, meta["gold_answer"])
    if clazz == "B":
        return score_code(completion, meta.get("test_imports", []), meta["test_list"])
    if clazz == "C":
        return score_instruction(completion, meta["check"])
    raise ValueError(clazz)
