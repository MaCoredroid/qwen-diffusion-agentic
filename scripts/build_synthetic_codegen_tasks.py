#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT = ROOT / "data/codegen_eval/synthetic_codegen_10.jsonl"
DEFAULT_SYSTEM = "You are a precise coding assistant."


TASKS = [
    {
        "id": "synthetic-codegen-00001",
        "task": "slugify_text",
        "entrypoint": "slugify_text",
        "prompt": (
            "Write a Python function `slugify_text(text)` that lowercases text, "
            "keeps ASCII letters and digits, replaces each run of other characters "
            "with one hyphen, and strips leading/trailing hyphens."
        ),
        "tests": [
            "assert slugify_text('Hello, World!') == 'hello-world'",
            "assert slugify_text('  Qwen 3.5 / Diffusion  ') == 'qwen-3-5-diffusion'",
            "assert slugify_text('Already---clean') == 'already-clean'",
            "assert slugify_text('!!!') == ''",
        ],
    },
    {
        "id": "synthetic-codegen-00002",
        "task": "merge_intervals",
        "entrypoint": "merge_intervals",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that accepts a "
            "list of [start, end] integer pairs and returns merged non-overlapping "
            "intervals sorted by start. Touching intervals should merge."
        ),
        "tests": [
            "assert merge_intervals([[5, 7], [1, 3], [2, 4]]) == [[1, 4], [5, 7]]",
            "assert merge_intervals([[1, 2], [2, 5], [8, 9]]) == [[1, 5], [8, 9]]",
            "assert merge_intervals([]) == []",
            "assert merge_intervals([[3, 3]]) == [[3, 3]]",
        ],
    },
    {
        "id": "synthetic-codegen-00003",
        "task": "top_k_frequent_words",
        "entrypoint": "top_k_frequent_words",
        "prompt": (
            "Write a Python function `top_k_frequent_words(words, k)` that returns "
            "the k most frequent words. Sort by descending frequency, then "
            "alphabetically ascending for ties."
        ),
        "tests": [
            "assert top_k_frequent_words(['b', 'a', 'b', 'c', 'a', 'b'], 2) == ['b', 'a']",
            "assert top_k_frequent_words(['z', 'x', 'z', 'x', 'a'], 3) == ['x', 'z', 'a']",
            "assert top_k_frequent_words([], 3) == []",
            "assert top_k_frequent_words(['same'], 5) == ['same']",
        ],
    },
    {
        "id": "synthetic-codegen-00004",
        "task": "parse_env_lines",
        "entrypoint": "parse_env_lines",
        "prompt": (
            "Write a Python function `parse_env_lines(text)` that parses KEY=VALUE "
            "lines into a dict. Ignore blank lines and lines whose first non-space "
            "character is '#'. Strip whitespace around keys and values. Split only "
            "on the first '='. Ignore lines without '='."
        ),
        "tests": [
            "assert parse_env_lines('A=1\\nB = two\\n# nope\\nEMPTY=') == {'A': '1', 'B': 'two', 'EMPTY': ''}",
            "assert parse_env_lines('TOKEN=a=b=c\\n bad line \\n X = y ') == {'TOKEN': 'a=b=c', 'X': 'y'}",
            "assert parse_env_lines('   # comment\\n\\n') == {}",
        ],
    },
    {
        "id": "synthetic-codegen-00005",
        "task": "balanced_brackets",
        "entrypoint": "balanced_brackets",
        "prompt": (
            "Write a Python function `balanced_brackets(s)` that returns True when "
            "the characters (), [], and {} are balanced and properly nested. Ignore "
            "all other characters."
        ),
        "tests": [
            "assert balanced_brackets('a(b[c]{d})') is True",
            "assert balanced_brackets('([)]') is False",
            "assert balanced_brackets('no brackets') is True",
            "assert balanced_brackets('((missing)') is False",
        ],
    },
    {
        "id": "synthetic-codegen-00006",
        "task": "word_wrap",
        "entrypoint": "word_wrap",
        "prompt": (
            "Write a Python function `word_wrap(text, width)` that wraps words into "
            "a list of lines with length at most width when possible. Do not split "
            "words. Collapse runs of whitespace between words. If a single word is "
            "longer than width, put it on its own line."
        ),
        "tests": [
            "assert word_wrap('alpha beta gamma', 10) == ['alpha beta', 'gamma']",
            "assert word_wrap('  many   spaces here ', 8) == ['many', 'spaces', 'here']",
            "assert word_wrap('superlongword tiny', 5) == ['superlongword', 'tiny']",
            "assert word_wrap('', 4) == []",
        ],
    },
    {
        "id": "synthetic-codegen-00007",
        "task": "multiset_added_lines",
        "entrypoint": "multiset_added_lines",
        "prompt": (
            "Write a Python function `multiset_added_lines(old, new)` that compares "
            "two newline-separated strings and returns the lines added in `new` "
            "relative to `old`, preserving their order in `new` and respecting "
            "duplicate line counts."
        ),
        "tests": [
            "assert multiset_added_lines('a\\nb\\n', 'a\\nb\\nc\\n') == ['c']",
            "assert multiset_added_lines('x\\nx\\ny', 'x\\ny\\nx\\nx') == ['x']",
            "assert multiset_added_lines('', 'one\\ntwo') == ['one', 'two']",
            "assert multiset_added_lines('same', 'same') == []",
        ],
    },
    {
        "id": "synthetic-codegen-00008",
        "task": "redact_secrets",
        "entrypoint": "redact_secrets",
        "prompt": (
            "Write a Python function `redact_secrets(config)` that returns a new "
            "dict with the same keys. If a key contains token, password, secret, or "
            "api_key case-insensitively, replace its value with '***REDACTED***'. "
            "Other values stay unchanged."
        ),
        "tests": [
            "src = {'api_key': 'abc', 'Name': 'demo', 'PASSWORD': 'pw'}; out = redact_secrets(src); assert out == {'api_key': '***REDACTED***', 'Name': 'demo', 'PASSWORD': '***REDACTED***'}",
            "assert redact_secrets({'accessToken': 't', 'nested_secret_value': 3}) == {'accessToken': '***REDACTED***', 'nested_secret_value': '***REDACTED***'}",
            "original = {'safe': 'x'}; result = redact_secrets(original); assert result == {'safe': 'x'} and result is not original",
        ],
    },
    {
        "id": "synthetic-codegen-00009",
        "task": "stable_dedupe",
        "entrypoint": "stable_dedupe",
        "prompt": (
            "Write a Python function `stable_dedupe(items)` that returns a list "
            "with duplicates removed while preserving first occurrence order. Items "
            "may be unhashable lists or dicts, so equality comparison must still work."
        ),
        "tests": [
            "assert stable_dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert stable_dedupe([[1], [1], [2]]) == [[1], [2]]",
            "assert stable_dedupe([{'a': 1}, {'a': 1}, {'b': 2}]) == [{'a': 1}, {'b': 2}]",
            "assert stable_dedupe([]) == []",
        ],
    },
    {
        "id": "synthetic-codegen-00010",
        "task": "apply_patch_ops",
        "entrypoint": "apply_patch_ops",
        "prompt": (
            "Write a Python function `apply_patch_ops(text, ops)` that applies a "
            "list of operations to a string. Each operation is a dict with `op` set "
            "to 'replace', 'insert', or 'delete'. Replace uses keys old/new and "
            "replaces all occurrences. Insert uses index/value and inserts before "
            "that character index, clamped to [0, len(text)]. Delete uses start/end "
            "and removes that slice, with bounds clamped."
        ),
        "tests": [
            "ops = [{'op': 'replace', 'old': 'cat', 'new': 'dog'}, {'op': 'insert', 'index': 0, 'value': 'A '}]; assert apply_patch_ops('cat sat', ops) == 'A dog sat'",
            "assert apply_patch_ops('abcdef', [{'op': 'delete', 'start': 2, 'end': 4}]) == 'abef'",
            "assert apply_patch_ops('abc', [{'op': 'insert', 'index': 99, 'value': '!'}]) == 'abc!'",
            "assert apply_patch_ops('abc', [{'op': 'delete', 'start': -5, 'end': 1}]) == 'bc'",
        ],
    },
]


def make_case(task):
    instruction = (
        "Return only Python code. Define the requested function exactly as named. "
        "Do not include prose, markdown fences, imports, file I/O, network calls, "
        "or example usage."
    )
    return {
        "source": "synthetic_codegen",
        "id": task["id"],
        "task": task["task"],
        "language": "python",
        "entrypoint": task["entrypoint"],
        "prompt_messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": task["prompt"]},
        ],
        "teacher_instruction": instruction,
        "tests": task["tests"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cases = [make_case(task) for task in TASKS]
    with args.out.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(args.out),
        "num_examples": len(cases),
        "language": "python",
        "entrypoints": [case["entrypoint"] for case in cases],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
