#!/usr/bin/env python3
"""Emit the Stage-A qwen-code smoke toy-repo case (planted bug + unittest).

One tiny zero-dependency Python project. `average()` has a single planted bug
(an extraneous ``+ 1``) that makes the seed unittest FAIL; the agent must read
the file, make a one-line edit, and re-run the test. Consumed by the proven
scripts/eval_qwen_code_repo_edit_cases.py harness (case schema: id/task/files/
test_command/expected_files/prompt). The SAME case is run against both arms
(diffusion :9952 and stock-AR :9951) so the arms are directly comparable.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "toy_repo_case.jsonl"

MATHUTILS = '''\
"""Tiny numeric helpers for the Stage-A qwen-code agentic smoke."""


def total(numbers):
    """Return the sum of a list of numbers."""
    result = 0
    for n in numbers:
        result += n
    return result


def average(numbers):
    """Return the arithmetic mean of a non-empty list of numbers."""
    return total(numbers) / len(numbers) + 1
'''

TEST = '''\
import unittest

from mathutils import average, total


class TestMathUtils(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total([2, 4, 6]), 12)

    def test_average_three(self):
        self.assertEqual(average([2, 4, 6]), 4)

    def test_average_single(self):
        self.assertEqual(average([10]), 10)

    def test_average_negatives(self):
        self.assertEqual(average([-3, 3]), 0)


if __name__ == "__main__":
    unittest.main()
'''

README = '''\
# tinycalc

Tiny numeric-helpers package used as the Stage-A qwen-code agentic smoke.

Run the tests with:

    python3 -m unittest test_mathutils
'''

case = {
    "id": "average_off_by_one",
    "task": "fix-buggy-average",
    "source": "stage_a_smoke",
    "files": {
        "mathutils.py": MATHUTILS,
        "test_mathutils.py": TEST,
        "README.md": README,
    },
    "test_command": "python3 -m unittest test_mathutils",
    "expected_files": ["mathutils.py"],
    "prompt": (
        "The average(numbers) function in mathutils.py returns a value that is "
        "too large by one. Read the file, find the bug, and fix mathutils.py so "
        "that average returns the correct arithmetic mean. Do not edit the tests."
    ),
}

OUT.write_text(json.dumps(case) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
