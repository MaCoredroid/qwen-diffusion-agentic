#!/usr/bin/env python3
"""Unit tests for VllmFlareEngineAdapter.run_turn against a MOCKED engine (no GPU).

Verifies Blocker C's turn-driver seam: prompt-token plumbing, output_ids assembly
(prompt + generated), FLARE stats -> reference-schema metric mapping, greedy
SamplingParams, and the schedule-events the audit battery consumes.

Run: .venv-vllm-p2-main/bin/python -m pytest scripts/test_parity_flare_engine_adapter.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path("/home/mark/qwen_diffusion")
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import parity_audit_flare_engine as P  # noqa: E402


class _FakeOut:
    def __init__(self, token_ids, text, finish_reason):
        self.token_ids = token_ids
        self.text = text
        self.finish_reason = finish_reason


class _FakeReqOut:
    def __init__(self, out):
        self.outputs = [out]


class _FakeModelState:
    def __init__(self, stats):
        self._stats = stats

    def stats(self):
        return self._stats


class FakeEngine:
    """Mimics the vLLM LLM seam the adapter uses: .generate(prompt, sp) and a
    FLARE ModelState reachable via .flare_model_state.stats()."""

    def __init__(self, gen_ids, text, finish_reason, stats):
        self._gen_ids = gen_ids
        self._text = text
        self._fr = finish_reason
        self.flare_model_state = _FakeModelState(stats)
        self.calls = []

    def generate(self, prompt, sampling_params):
        self.calls.append((prompt, sampling_params))
        return [_FakeReqOut(_FakeOut(self._gen_ids, self._text, self._fr))]


def _ctx(temperature=0.0, max_new_tokens=64, stop=(248044,)):
    return P.TurnContext(
        model=None,
        tokenizer=None,
        prompt_input_ids=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
        block_size=32,
        max_new_tokens=max_new_tokens,
        mask_id=248077,
        stop_token_ids=set(stop),
        top_p=0.95,
        temperature=temperature,
        schemas={},
        grammar_topk=256,
    )


def _hybrid_stats():
    return {
        "block_size": 32,
        "decode_mode": "hybrid_clean",
        "read_calls": 10,
        "advance_calls": 1,
        "read_advance_ratio": 10.0,
        "route_verified": False,
        "hybrid_clean": {
            "model_forwards": 7,
            "forced_token_count": 19,
            "zero_forward_rows": 3,
            "value_tokens": 4,
            "projected_value_tokens_exact": 0,
            "tokens_per_forward": 3.71,
        },
    }


def test_run_turn_output_ids_and_metric_mapping():
    fake = FakeEngine([101, 102, 103], '{"a":1}', "stop", _hybrid_stats())
    ad = P.VllmFlareEngineAdapter(engine=fake)
    ad.preflight()  # injected engine -> must not raise
    res = ad.run_turn(_ctx())

    assert res.output_ids == [5, 6, 7, 8, 101, 102, 103]
    assert res.block_snapshots == []
    m = res.metrics
    assert m["denoise_forwards"] == 7
    assert m["forwards_per_turn"] == 7.0
    assert m["forced_grammar_tokens"] == 19
    assert m["model_value_tokens"] == 4
    assert m["model_structural_tokens"] == 3  # model_forwards(7) - value_tokens(4)
    assert m["grammar_replacement_value_tokens"] == 0
    assert m["stop_reason"] == "complete_tool_call"
    assert m["decode_mode"] == "hybrid_clean"
    assert m["read_advance_ratio"] == 10.0
    assert m["block_size"] == 32
    assert isinstance(m["wall_seconds"], float)


def test_run_turn_prompt_and_sampling_params():
    fake = FakeEngine([9], "x", "stop", _hybrid_stats())
    ad = P.VllmFlareEngineAdapter(engine=fake)
    ad.run_turn(_ctx(temperature=0.0, max_new_tokens=64))
    prompt, sp = fake.calls[0]
    assert prompt == {"prompt_token_ids": [5, 6, 7, 8]}
    assert sp.max_tokens == 64
    assert sp.temperature == 0.0
    assert sp.top_p == 1.0  # greedy collapses top_p
    assert 248044 in set(sp.stop_token_ids)


def test_run_turn_sampled_keeps_top_p():
    fake = FakeEngine([9], "x", "length", _hybrid_stats())
    ad = P.VllmFlareEngineAdapter(engine=fake)
    ad.run_turn(_ctx(temperature=0.7))
    _, sp = fake.calls[0]
    assert sp.temperature == 0.7
    assert abs(sp.top_p - 0.95) < 1e-9


def test_schedule_events_consumed_by_audit_battery():
    fake = FakeEngine([101, 102], "x", "stop", _hybrid_stats())
    ad = P.VllmFlareEngineAdapter(engine=fake)
    res = ad.run_turn(_ctx())
    row = P.build_audit_row(
        backend="engine_vllm",
        generated_ids=res.output_ids[4:],
        assistant_text="x",
        exact_arguments=False,
        metrics=res.metrics,
    )
    ev = row["backend_meta"]["sampler_schedule_events"]
    assert ev["denoise_forwards_total"] == 7
    assert ev["hybrid_model_forwards"] == 7
    assert ev["hybrid_forced_grammar_tokens"] == 19
    assert ev["hybrid_model_value_tokens"] == 4
    assert ev["hybrid_model_structural_tokens"] == 3
    assert ev["hybrid_grammar_replacement_value_tokens"] == 0
    # zero-value-projection gate reads this key -> must be 0 for a clean turn
    assert P.grammar_value_projection_count(res.metrics) == 0


def test_canvas_mode_denoise_falls_back_to_read_calls():
    stats = {"block_size": 32, "decode_mode": "canvas", "read_calls": 9, "advance_calls": 1}
    fake = FakeEngine([1, 2], "..", "length", stats)
    ad = P.VllmFlareEngineAdapter(engine=fake)
    res = ad.run_turn(_ctx())
    assert res.metrics["denoise_forwards"] == 9
    assert res.metrics["stop_reason"] == "max_new_tokens"
    assert res.metrics["decode_mode"] == "canvas"


def test_missing_stats_falls_back_to_generated_len():
    class NoStatsEngine(FakeEngine):
        def __init__(self):
            super().__init__([1, 2, 3, 4], "y", "length", {})
            del self.flare_model_state  # no stats seam at all

    fake = NoStatsEngine()
    ad = P.VllmFlareEngineAdapter(engine=fake)
    res = ad.run_turn(_ctx())
    # no hybrid_clean, no read_calls -> denoise defaults to len(generated)
    assert res.metrics["denoise_forwards"] == 4
    assert "stats_error" in res.metrics["engine_stats"]


def test_preflight_requires_cuda_without_engine(monkeypatch):
    ad = P.VllmFlareEngineAdapter(model_path=None)  # no injected engine
    monkeypatch.setattr(P.torch.cuda, "is_available", lambda: False)
    with pytest.raises(P.EngineUnavailable):
        ad.preflight()


def test_snapshot_from_modelstate_is_explicit_followon():
    ad = P.VllmFlareEngineAdapter(engine=FakeEngine([1], "z", "stop", _hybrid_stats()))
    with pytest.raises(P.EngineUnavailable):
        ad.snapshot_from_vllm_modelstate(object(), block_start=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
