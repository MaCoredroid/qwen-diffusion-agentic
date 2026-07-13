#!/usr/bin/env python3
"""Unit tests for the HARNESS TRUTH-TELLING ctx-overflow classification (2026-07-12).

Covers the end-to-end mapping the fix wires so a context-window cap-death (the
proxy's context-overflow retry ladder exhausts -> vLLM's terminal HTTP 400 is
surfaced as the episode's `result` payload; the 32516/253/32769 cap+1 signature
documented in runs/k_gate_c46/K1_COMMITTAL_ANALYSIS.md "Terminal trigger") produces
a DISTINGUISHABLE record instead of being mislabeled as an honest empty-patch miss /
clean exit-0 quit:

  1. DRIVER (run_swe_bench_qwen_code._classify_terminal_cause): a mocked cap-death
     episode record -> "ctx_overflow"; a normal/loop-halt record -> None; the
     empty-retry fallback (final re-drive produced no terminal text) still resolves
     to the main drive's ctx-overflow terminal.
  2. LEDGER (ledger._classify / _ctx_overflow_ids / cmd_record): a ctx_overflow +
     empty-patch id -> "env_limited" (NOT empty_patch); a genuine empty -> empty_patch;
     resolved/error/unresolved unaffected; and a full cmd_record over a mocked
     batchdir writes the env_limited verdict with a terminal_cause=ctx_overflow stamp.
  3. NO retroactive surgery: a batchdir whose runner_metadata predate the tag yields
     an empty ctx-overflow set -> old empty_patch labeling preserved.

No network, no GPU, no docker. Run: python3 scripts/test_terminal_cause_classification.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "runs" / "swe_datagen_s1"))

import run_swe_bench_qwen_code as D  # driver
import ledger as L  # datagen ledger


# The byte-exact terminal payload qwen-code surfaces when the retry ladder bottoms
# out at max_tokens=253 with the prompt pinned at the 32516 cap (32516+253=32769=cap+1).
CAP_DEATH_RESULT_TAIL = (
    "[API Error: 400 This model's maximum context length is 32768 tokens. However, "
    "you requested 253 output tokens and your prompt contains at least 32516 input "
    "tokens, for a total of at least 32769 tokens. Please reduce the length of the "
    "input prompt or the number of requested output tokens. "
    "(parameter=input_tokens, value=32516)]"
)


def _cap_death_meta(*, with_empty_retry=False):
    """A mocked runner_metadata summary for a context-window cap-death episode
    (matches the real C46 shape: exit 0, subtype 'success', patch_bytes 0, the 400
    signature in the terminal result)."""
    m = {
        "instance_id": "django__django-12273",
        "patch_bytes": 0,
        "qwen": {
            "exit_code": 0, "timed_out": False, "subtype": "success", "num_turns": 5,
            "result_tail": CAP_DEATH_RESULT_TAIL,
        },
    }
    if with_empty_retry:
        # The empty-patch re-drive that produced NO terminal text (real C46: 6/41
        # cap-deaths have an empty retry result_tail) — must fall back to the main
        # drive's ctx-overflow terminal, not be read as a non-ctx (None) terminal.
        m["qwen_retry1"] = {"exit_code": 0, "num_turns": 0, "result_tail": ""}
        m["empty_patch_retry"] = {"cause": "agent_gave_up", "max_retries": 1,
                                  "recovered_patch_bytes": 0}
    return m


class DriverTerminalCause(unittest.TestCase):
    def test_cap_death_is_ctx_overflow(self):
        m = _cap_death_meta()
        self.assertEqual(D._classify_terminal_cause(D._agent_metas(m)),
                         D.TERMINAL_CTX_OVERFLOW)

    def test_cap_death_with_empty_retry_falls_back_to_main(self):
        m = _cap_death_meta(with_empty_retry=True)
        # The last attempt produced no terminal text -> skip it -> main drive's
        # ctx-overflow terminal is the episode's true terminal event.
        self.assertEqual(D._classify_terminal_cause(D._agent_metas(m)),
                         D.TERMINAL_CTX_OVERFLOW)

    def test_normal_summary_is_not_ctx_overflow(self):
        m = {"qwen": {"exit_code": 0, "result_tail": "I edited views.py and the "
                      "failing test now passes. Done."}}
        self.assertIsNone(D._classify_terminal_cause(D._agent_metas(m)))

    def test_loop_halt_is_not_ctx_overflow(self):
        m = {"qwen": {"exit_code": 1, "result_tail": "loop detected: repeated tool "
                      "call run_shell_command"}}
        self.assertIsNone(D._classify_terminal_cause(D._agent_metas(m)))

    def test_quoted_phrase_without_api_error_is_not_flagged(self):
        # A model that merely MENTIONS the phrase in prose (no API-error framing)
        # must NOT be misread as an env-limited death.
        self.assertFalse(D._result_is_ctx_overflow(
            "The docstring notes the maximum context length is 32768 tokens for "
            "this model, which I summarized in the report."))

    def test_ctx_overflow_requires_the_phrase(self):
        self.assertFalse(D._result_is_ctx_overflow("[API Error: 400 rate limited]"))
        self.assertFalse(D._result_is_ctx_overflow(""))
        self.assertFalse(D._result_is_ctx_overflow(None))

    def test_retry_that_recovers_non_ctx_is_not_ctx_overflow(self):
        # Main drove into ctx-overflow, but the re-drive TERMINATED with a normal
        # summary -> the episode's terminal event is the re-drive, not a cap-death.
        m = {"qwen": {"exit_code": 0, "result_tail": CAP_DEATH_RESULT_TAIL},
             "qwen_retry1": {"exit_code": 0, "result_tail": "Fixed it; tests pass."}}
        self.assertIsNone(D._classify_terminal_cause(D._agent_metas(m)))


class LedgerClassify(unittest.TestCase):
    def test_ctx_overflow_empty_maps_to_env_limited(self):
        v = L._classify("a", resolved=set(), empty={"a"}, error=set(),
                        unresolved=set(), pull_failed=set(), ctx_overflow={"a"})
        self.assertEqual(v, "env_limited")

    def test_genuine_empty_stays_empty_patch(self):
        v = L._classify("b", resolved=set(), empty={"b"}, error=set(),
                        unresolved=set(), pull_failed=set(), ctx_overflow=set())
        self.assertEqual(v, "empty_patch")

    def test_resolved_error_unresolved_unaffected_by_ctx_flag(self):
        # A ctx-overflow tag NEVER overrides a scoreable outcome: only the empty
        # bucket (the mislabeled honest-miss) is rerouted.
        self.assertEqual(L._classify("r", {"r"}, set(), set(), set(), set(), {"r"}),
                         "resolved")
        self.assertEqual(L._classify("e", set(), set(), {"e"}, set(), set(), {"e"}),
                         "error")
        self.assertEqual(L._classify("u", set(), set(), set(), {"u"}, set(), {"u"}),
                         "unresolved")

    def test_env_limited_is_a_real_verdict(self):
        # yield/coverage-neutral vs the empty_patch it replaces (both REAL, non-resolved).
        self.assertIn("env_limited", L.REAL_VERDICTS)


class LedgerRecordEndToEnd(unittest.TestCase):
    def _mk_batch(self, tmp, *, tag_terminal_cause):
        """Build a minimal batchdir: subset + merged score report + per-task
        runner_metadata under gen/. tag_terminal_cause toggles whether the cap-death
        episode carries the driver's terminal_cause tag (prospective) or not
        (historical -> old labeling)."""
        bd = Path(tmp)
        ids = ["cap__death-1", "true__empty-1", "good__resolved-1"]
        (bd / "subset.json").write_text(json.dumps({"instance_ids": ids}))
        score = bd / "score"
        score.mkdir(parents=True)
        (score / "datagen-eval.b1.json").write_text(json.dumps({
            "resolved_ids": ["good__resolved-1"],
            "unresolved_ids": [],
            # BOTH cap-death and true-empty land in empty_patch_ids from the harness
            # (both produced an empty patch); the ledger disambiguates via terminal_cause.
            "empty_patch_ids": ["cap__death-1", "true__empty-1"],
            "error_ids": [],
        }))
        for iid in ids:
            pt = bd / "gen" / "shard_0" / "verified" / "per_task" / iid
            pt.mkdir(parents=True)
            meta = {"instance_id": iid}
            if iid == "cap__death-1" and tag_terminal_cause:
                meta["terminal_cause"] = "ctx_overflow"
            elif iid == "cap__death-1":
                meta["terminal_cause"] = None  # historical: field present but untagged
            (pt / "runner_metadata.json").write_text(json.dumps(meta))
        return bd, ids

    def _verdicts(self, attempts_path):
        return {json.loads(l)["instance_id"]: json.loads(l)
                for l in attempts_path.read_text().splitlines() if l.strip()}

    def test_prospective_cap_death_records_env_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            bd, _ = self._mk_batch(tmp, tag_terminal_cause=True)
            self.assertEqual(L._ctx_overflow_ids(bd), {"cap__death-1"})
            attempts = bd / "attempts.jsonl"
            rc = L.cmd_record(bd, "b1", attempts)
            self.assertEqual(rc, 0)
            rows = self._verdicts(attempts)
            self.assertEqual(rows["cap__death-1"]["verdict"], "env_limited")
            self.assertEqual(rows["cap__death-1"]["terminal_cause"], "ctx_overflow")
            self.assertEqual(rows["true__empty-1"]["verdict"], "empty_patch")
            self.assertNotIn("terminal_cause", rows["true__empty-1"])
            self.assertEqual(rows["good__resolved-1"]["verdict"], "resolved")

    def test_historical_batch_keeps_empty_patch(self):
        # NO retroactive surgery: an untagged (historical) cap-death stays empty_patch.
        with tempfile.TemporaryDirectory() as tmp:
            bd, _ = self._mk_batch(tmp, tag_terminal_cause=False)
            self.assertEqual(L._ctx_overflow_ids(bd), set())
            attempts = bd / "attempts.jsonl"
            L.cmd_record(bd, "b1", attempts)
            rows = self._verdicts(attempts)
            self.assertEqual(rows["cap__death-1"]["verdict"], "empty_patch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
