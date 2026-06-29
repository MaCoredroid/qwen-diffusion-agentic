#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT = ROOT / "data/repo_edit_eval/tiny_repo_edit_5.jsonl"
DEFAULT_TEST_COMMAND = "python3 -m unittest discover -s tests -v"


TASKS = [
    {
        "id": "tiny-repo-edit-001",
        "task": "slugify_text",
        "prompt": (
            "Fix the slugify_text function so all tests pass. Keep the public "
            "function name and do not add third-party dependencies."
        ),
        "files": {
            "text_tools.py": '''
def slugify_text(value):
    """Return a URL-safe slug for value."""
    return value.lower().replace(" ", "-")
''',
            "tests/test_text_tools.py": '''
import unittest

from text_tools import slugify_text


class SlugifyTextTests(unittest.TestCase):
    def test_basic_cleanup(self):
        self.assertEqual(slugify_text("Hello, World!"), "hello-world")

    def test_collapses_separators(self):
        self.assertEqual(slugify_text("  Multiple___Spaces---Here  "), "multiple-spaces-here")

    def test_keeps_digits(self):
        self.assertEqual(slugify_text("Release 2026.06"), "release-2026-06")


if __name__ == "__main__":
    unittest.main()
''',
        },
        "expected_files": ["text_tools.py"],
    },
    {
        "id": "tiny-repo-edit-002",
        "task": "merge_intervals",
        "prompt": (
            "Fix merge_intervals so adjacent and overlapping intervals are merged "
            "and the output is sorted. Do not change the function signature."
        ),
        "files": {
            "intervals.py": '''
def merge_intervals(intervals):
    ordered = sorted(intervals)
    merged = []
    for start, end in ordered:
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return [tuple(item) for item in merged]
''',
            "tests/test_intervals.py": '''
import unittest

from intervals import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_merges_overlap_and_adjacency(self):
        self.assertEqual(merge_intervals([(5, 7), (1, 3), (3, 5)]), [(1, 7)])

    def test_keeps_gaps(self):
        self.assertEqual(merge_intervals([(10, 12), (1, 2), (4, 8)]), [(1, 2), (4, 8), (10, 12)])

    def test_extends_to_max_end(self):
        self.assertEqual(merge_intervals([(1, 10), (2, 3), (9, 14)]), [(1, 14)])


if __name__ == "__main__":
    unittest.main()
''',
        },
        "expected_files": ["intervals.py"],
    },
    {
        "id": "tiny-repo-edit-003",
        "task": "parse_env_lines",
        "prompt": (
            "Fix parse_env_lines. It should parse KEY=VALUE lines, ignore blanks "
            "and comments, strip whitespace around keys and values, and preserve "
            "equals signs inside values."
        ),
        "files": {
            "envparse.py": '''
def parse_env_lines(lines):
    values = {}
    for line in lines:
        key, value = line.split("=")
        values[key] = value
    return values
''',
            "tests/test_envparse.py": '''
import unittest

from envparse import parse_env_lines


class ParseEnvLinesTests(unittest.TestCase):
    def test_ignores_comments_and_blanks(self):
        self.assertEqual(parse_env_lines(["# comment", "", "PORT=8080"]), {"PORT": "8080"})

    def test_strips_whitespace(self):
        self.assertEqual(parse_env_lines([" HOST = localhost "]), {"HOST": "localhost"})

    def test_preserves_equals_in_value(self):
        self.assertEqual(parse_env_lines(["TOKEN=a=b=c"]), {"TOKEN": "a=b=c"})


if __name__ == "__main__":
    unittest.main()
''',
        },
        "expected_files": ["envparse.py"],
    },
    {
        "id": "tiny-repo-edit-004",
        "task": "redact_secrets",
        "prompt": (
            "Fix redact_secrets so it redacts values for sensitive keys in URLs "
            "and assignment-like strings. Keep non-sensitive fields unchanged."
        ),
        "files": {
            "redact.py": '''
def redact_secrets(text):
    sensitive = ["token", "password", "secret"]
    parts = text.split("&")
    output = []
    for part in parts:
        if "=" not in part:
            output.append(part)
            continue
        key, value = part.split("=", 1)
        if key in sensitive:
            value = "***"
        output.append(key + "=" + value)
    return "&".join(output)
''',
            "tests/test_redact.py": '''
import unittest

from redact import redact_secrets


class RedactSecretsTests(unittest.TestCase):
    def test_url_query_case_insensitive(self):
        self.assertEqual(redact_secrets("user=mark&Token=abc123&debug=true"), "user=mark&Token=***&debug=true")

    def test_assignment_style(self):
        self.assertEqual(redact_secrets("password = hunter2"), "password = ***")

    def test_substring_not_sensitive(self):
        self.assertEqual(redact_secrets("secretary=Jane&secret=real"), "secretary=Jane&secret=***")


if __name__ == "__main__":
    unittest.main()
''',
        },
        "expected_files": ["redact.py"],
    },
    {
        "id": "tiny-repo-edit-005",
        "task": "word_wrap",
        "prompt": (
            "Fix word_wrap so it returns lines whose lengths are at most width, "
            "wrapping on spaces when possible. Words longer than width may stay "
            "on their own line."
        ),
        "files": {
            "wraptext.py": '''
def word_wrap(text, width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width:
            lines.append(candidate)
            current = ""
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
''',
            "tests/test_wraptext.py": '''
import unittest

from wraptext import word_wrap


class WordWrapTests(unittest.TestCase):
    def test_wraps_on_spaces(self):
        self.assertEqual(word_wrap("alpha beta gamma delta", 11), ["alpha beta", "gamma delta"])

    def test_no_line_exceeds_width_when_possible(self):
        lines = word_wrap("one two three four", 8)
        self.assertEqual(lines, ["one two", "three", "four"])
        self.assertTrue(all(len(line) <= 8 for line in lines))

    def test_long_word_stays_alone(self):
        self.assertEqual(word_wrap("tiny extraordinary word", 6), ["tiny", "extraordinary", "word"])


if __name__ == "__main__":
    unittest.main()
''',
        },
        "expected_files": ["wraptext.py"],
    },
]


def clean_multiline(text):
    return text.strip("\n") + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-command", default=DEFAULT_TEST_COMMAND)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for task in TASKS:
            row = {
                "source": "synthetic_repo_edit",
                "id": task["id"],
                "task": task["task"],
                "prompt": task["prompt"],
                "test_command": args.test_command,
                "expected_files": task["expected_files"],
                "files": {path: clean_multiline(content) for path, content in task["files"].items()},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "out": str(args.out),
        "records": len(TASKS),
        "test_command": args.test_command,
        "ids": [task["id"] for task in TASKS],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
