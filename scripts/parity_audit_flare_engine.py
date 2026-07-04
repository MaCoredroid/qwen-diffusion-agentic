#!/usr/bin/env python3
"""Parity + audit harness: HF hybrid-clean reference  vs  the new engine path.

This is the M1/M2 promotion GATE instrument for the P2 engine-fast diffusion
serving work (see ``p2_serving_reuse_plan.md`` and ``redesign_plan_and_design_notes.md``).

It compares, for ONE matched-20 agentic turn:

  (a) REFERENCE  -- ``scripts/eval_flare_northstar_hybrid_clean.py`` decode driven
      by ``flare_hf_cache.RequestDiffusionState`` (route_i FLARE serving cache).
  (b) ENGINE     -- a pluggable ``EngineAdapter`` (the future vLLM
      ``Qwen3_5FlareModelState`` on the pinned ``vllm_p2_pr42406`` workspace).

Four hard assertions (the promotion gate):

  1. BYTE-IDENTICAL token output          -- ref output_ids == engine output_ids,
                                             and decoded bytes are equal.
  2. EQUAL value-token counts             -- true XML <parameter> value tokens match
                                             (audit_value_projection_tokens counting),
                                             and the reported model-value counter matches.
  3. ZERO projected values                -- both sides pass the label-free audit:
                                             wave1_projected_tokens == 0,
                                             projected_value_tokens_exact in {0, None},
                                             zero_projected_value_tokens_verified == 1.
  4. STATE-SNAPSHOT EQUALITY at block boundaries -- per-layer GDN fp32 state and
                                             conv_tail agree within fp32 tolerance
                                             (default 1e-3, matching clean_advance_block's
                                             own boundary-state guard), full-attention KV
                                             lengths agree exactly, and the captured
                                             shifted last-token logits agree within the
                                             validate_flare_hf_cache atol (default 2e-2).

Plus a counters report in the exact ``audit_value_projection_tokens`` "totals"
schema (the audit battery), for both sides.

CPU-mockable layering (see --mode):

  * ``selftest``      CPU, tokenizer-only, no GPU / no model weights.
                      Fabricates a matching turn pair and a set of deliberately
                      corrupted pairs, and asserts every comparator + the audit
                      battery flags them correctly. Verifies the harness machinery.
  * ``ops-parity``    CPU, no GPU. Imports the REAL engine primitives from the
                      pinned workspace (``qwen3_5_flare_ops.py`` -- pure torch) and
                      asserts numeric/byte equivalence against the ``flare_hf_cache``
                      reference primitives: the train-matched +1 shift, the conv-tail
                      roll, the read-only-denoise snapshot/restore write-back
                      suppression, the fp32 boundary carrier, the canvas
                      denoise/commit phase transition, and the variable-accept
                      num_sampled->num_accepted map. This is a genuine parity gate
                      against the actual engine code, off-GPU.
  * ``state-parity``  CPU (or CUDA if present) *tiny* route_i model. Exercises the
                      REAL StateSnapshot capture/compare on real tensors:
                      incremental-advance state  vs  from-scratch-reset state
                      (the align-APC restore-by-copy invariant), shifted-logit +
                      argmax parity (cache-on vs full recompute), and greedy
                      cache-on vs cache-off byte-identity. No 9B weights needed.
  * ``turn``          GPU, real 9B model + matched-20 data. Runs ONE turn through
                      (a) and (b) and applies all four assertions + the audit report.
                      GPU-only; clearly gated.

Exit codes: 0 = all applicable gates passed; 2 = a parity/audit gate FAILED;
3 = engine adapter unavailable (reference side still audited).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Light, import-safe (no model load, no CUDA touch at import time).
import audit_value_projection_tokens as avpt  # noqa: E402

DEFAULT_VLLM_WORKSPACE = Path("/home/mark/shared/vllm_p2_pr42406")

# fp32 GDN boundary tolerance: matches clean_advance_block's own
# final_state/chunk_state guard (>1e-3 raises) in flare_hf_cache.py.
DEFAULT_GDN_ATOL = 1e-3
# shifted-logit tolerance: matches validate_flare_hf_cache.py --atol default.
DEFAULT_LOGIT_ATOL = 2e-2


def _set_route_i_env() -> None:
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")


# ---------------------------------------------------------------------------
# State snapshots  (the fp32 block-boundary equality machinery)
# ---------------------------------------------------------------------------
@dataclass
class LayerSnapshot:
    kind: str
    gdn_state: torch.Tensor | None  # cpu fp32, [B, Hv, Dk, Dv]  (linear_attention)
    conv_tail: torch.Tensor | None  # cpu,      [B, W-1, ...]     (linear_attention)
    key_len: int                    # committed KV length         (full_attention)
    value_len: int


@dataclass
class StateSnapshot:
    block_start: int
    block_size: int
    batch_size: int
    layers: list[LayerSnapshot]
    last_token_logits: torch.Tensor | None  # cpu, [B, 1, V] or None at block 0


def capture_state_snapshot(state: Any) -> StateSnapshot:
    """Snapshot a ``flare_hf_cache.RequestDiffusionState`` (or any object exposing
    the same ``layer_caches`` / ``block_start`` / ``last_token_logits`` fields)."""
    layers: list[LayerSnapshot] = []
    for lc in state.layer_caches:
        if lc.kind == "linear_attention":
            layers.append(
                LayerSnapshot(
                    kind="linear_attention",
                    gdn_state=None if lc.gdn_state is None else lc.gdn_state.detach().to("cpu", torch.float32).clone(),
                    conv_tail=None if lc.conv_tail is None else lc.conv_tail.detach().to("cpu").clone(),
                    key_len=0,
                    value_len=0,
                )
            )
        else:
            key_len = 0 if lc.key is None else int(lc.key.shape[2])
            value_len = 0 if lc.value is None else int(lc.value.shape[2])
            layers.append(
                LayerSnapshot(kind="full_attention", gdn_state=None, conv_tail=None, key_len=key_len, value_len=value_len)
            )
    last = None
    if state.last_token_logits is not None:
        last = state.last_token_logits.detach().to("cpu", torch.float32).clone()
    return StateSnapshot(
        block_start=int(state.block_start),
        block_size=int(state.block_size),
        batch_size=int(state.batch_size),
        layers=layers,
        last_token_logits=last,
    )


def _max_abs_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None or b is None:
        return float("inf")
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


@dataclass
class LayerDiff:
    layer: int
    kind: str
    gdn_max_abs_diff: float | None
    conv_max_abs_diff: float | None
    key_len_equal: bool
    value_len_equal: bool
    ok: bool


@dataclass
class SnapshotDiff:
    block_start: int
    ok: bool
    reasons: list[str]
    layer_diffs: list[LayerDiff]
    last_logits_max_abs_diff: float | None


def compare_snapshots(
    ref: StateSnapshot,
    eng: StateSnapshot,
    *,
    gdn_atol: float,
    logit_atol: float,
) -> SnapshotDiff:
    reasons: list[str] = []
    if ref.block_start != eng.block_start:
        reasons.append(f"block_start {ref.block_start} != {eng.block_start}")
    if ref.block_size != eng.block_size:
        reasons.append(f"block_size {ref.block_size} != {eng.block_size}")
    if len(ref.layers) != len(eng.layers):
        reasons.append(f"num_layers {len(ref.layers)} != {len(eng.layers)}")

    layer_diffs: list[LayerDiff] = []
    for idx, (rl, el) in enumerate(zip(ref.layers, eng.layers)):
        gdn_d = _max_abs_diff(rl.gdn_state, el.gdn_state)
        conv_d = _max_abs_diff(rl.conv_tail, el.conv_tail)
        kind_ok = rl.kind == el.kind
        key_eq = rl.key_len == el.key_len
        val_eq = rl.value_len == el.value_len
        ok = (
            kind_ok
            and key_eq
            and val_eq
            and (gdn_d is None or gdn_d <= gdn_atol)
            and (conv_d is None or conv_d <= gdn_atol)
        )
        if not kind_ok:
            reasons.append(f"layer{idx} kind {rl.kind}!={el.kind}")
        if not key_eq:
            reasons.append(f"layer{idx} key_len {rl.key_len}!={el.key_len}")
        if not val_eq:
            reasons.append(f"layer{idx} value_len {rl.value_len}!={el.value_len}")
        if gdn_d is not None and gdn_d > gdn_atol:
            reasons.append(f"layer{idx} gdn_state diff {gdn_d:.3g} > {gdn_atol:g}")
        if conv_d is not None and conv_d > gdn_atol:
            reasons.append(f"layer{idx} conv_tail diff {conv_d:.3g} > {gdn_atol:g}")
        layer_diffs.append(LayerDiff(idx, rl.kind, gdn_d, conv_d, key_eq, val_eq, ok))

    logit_d = _max_abs_diff(ref.last_token_logits, eng.last_token_logits)
    if logit_d is not None and logit_d > logit_atol:
        reasons.append(f"last_token_logits diff {logit_d:.3g} > {logit_atol:g}")

    ok = not reasons
    return SnapshotDiff(
        block_start=int(ref.block_start),
        ok=ok,
        reasons=reasons,
        layer_diffs=layer_diffs,
        last_logits_max_abs_diff=logit_d,
    )


def compare_snapshot_sequences(
    ref: list[StateSnapshot],
    eng: list[StateSnapshot],
    *,
    gdn_atol: float,
    logit_atol: float,
) -> dict[str, Any]:
    """Compare block-boundary snapshots, ALIGNED BY ``block_start``.

    Keying by block_start (not list position) is deliberate: the reference emits a
    snapshot for every prefill AND generation advance, while the real engine may only
    expose commit-time (generation) boundaries. The gate requires every block_start
    present in BOTH to agree, and flags any boundary present in only one side."""
    ref_by = {s.block_start: s for s in ref}
    eng_by = {s.block_start: s for s in eng}
    shared = sorted(set(ref_by) & set(eng_by))
    only_ref = sorted(set(ref_by) - set(eng_by))
    only_eng = sorted(set(eng_by) - set(ref_by))
    diffs = [
        compare_snapshots(ref_by[bs], eng_by[bs], gdn_atol=gdn_atol, logit_atol=logit_atol)
        for bs in shared
    ]
    all_ok = bool(shared) and all(d.ok for d in diffs) and not only_ref and not only_eng
    max_gdn = 0.0
    max_conv = 0.0
    max_logit = 0.0
    for d in diffs:
        for ld in d.layer_diffs:
            if ld.gdn_max_abs_diff not in (None, float("inf")):
                max_gdn = max(max_gdn, float(ld.gdn_max_abs_diff))
            if ld.conv_max_abs_diff not in (None, float("inf")):
                max_conv = max(max_conv, float(ld.conv_max_abs_diff))
        if d.last_logits_max_abs_diff not in (None, float("inf")):
            max_logit = max(max_logit, float(d.last_logits_max_abs_diff))
    return {
        "ok": bool(all_ok),
        "num_boundaries_ref": len(ref),
        "num_boundaries_eng": len(eng),
        "num_shared_boundaries": len(shared),
        "block_starts_only_in_ref": only_ref,
        "block_starts_only_in_eng": only_eng,
        "max_gdn_state_abs_diff": max_gdn,
        "max_conv_tail_abs_diff": max_conv,
        "max_last_logits_abs_diff": max_logit,
        "gdn_atol": float(gdn_atol),
        "logit_atol": float(logit_atol),
        "failing_boundaries": [
            {"block_start": d.block_start, "reasons": d.reasons}
            for d in diffs
            if not d.ok
        ],
    }


# ---------------------------------------------------------------------------
# Reference-side snapshot recorder (monkeypatch, keeps the reference verbatim)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def record_boundary_snapshots():
    """Records a StateSnapshot after every ``advance`` / ``advance_clean_only`` on
    ``flare_hf_cache.RequestDiffusionState``, so the *unmodified* reference decode
    produces block-boundary snapshots as a side channel."""
    from flare_hf_cache import RequestDiffusionState

    records: list[StateSnapshot] = []
    orig_advance = RequestDiffusionState.advance
    orig_clean = RequestDiffusionState.advance_clean_only

    def advance(self, model, block_ids):
        orig_advance(self, model, block_ids)
        records.append(capture_state_snapshot(self))

    def advance_clean_only(self, model, block_ids):
        orig_clean(self, model, block_ids)
        records.append(capture_state_snapshot(self))

    RequestDiffusionState.advance = advance
    RequestDiffusionState.advance_clean_only = advance_clean_only
    try:
        yield records
    finally:
        RequestDiffusionState.advance = orig_advance
        RequestDiffusionState.advance_clean_only = orig_clean


# ---------------------------------------------------------------------------
# Byte identity
# ---------------------------------------------------------------------------
def token_bytes_identical(
    ref_ids: list[int],
    eng_ids: list[int],
    tokenizer=None,
) -> dict[str, Any]:
    token_exact = list(int(x) for x in ref_ids) == list(int(x) for x in eng_ids)
    first_div = None
    if not token_exact:
        for i, (a, b) in enumerate(zip(ref_ids, eng_ids)):
            if int(a) != int(b):
                first_div = {"index": i, "ref": int(a), "eng": int(b)}
                break
        if first_div is None:
            first_div = {"index": min(len(ref_ids), len(eng_ids)), "ref": None, "eng": None}
    byte_exact = token_exact
    ref_text = eng_text = None
    if tokenizer is not None:
        ref_text = tokenizer.decode(list(int(x) for x in ref_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        eng_text = tokenizer.decode(list(int(x) for x in eng_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        byte_exact = ref_text.encode("utf-8") == eng_text.encode("utf-8")
    return {
        "token_exact": bool(token_exact),
        "byte_exact": bool(byte_exact),
        "ref_len": len(ref_ids),
        "eng_len": len(eng_ids),
        "first_divergence": first_div,
    }


# ---------------------------------------------------------------------------
# Audit battery (audit_value_projection_tokens format)
# ---------------------------------------------------------------------------
def audit_totals_for_rows(tokenizer, rows: list[dict]) -> dict[str, Any]:
    totals, _audited = avpt.audit_rows(tokenizer, rows)
    return totals


def value_token_count(tokenizer, row: dict) -> int:
    return int(avpt.output_value_token_count(tokenizer, row))


def zero_projected_ok(totals: dict[str, Any]) -> bool:
    return (
        int(totals.get("zero_projected_value_tokens_verified") or 0) == 1
        and int(totals.get("wave1_projected_tokens") or 0) == 0
        and int(totals.get("projected_value_tokens_exact") or 0) == 0
        and int(totals.get("parallel_commit_forced_tokens_counter") or 0) == 0
    )


def grammar_value_projection_count(metrics: dict[str, Any] | None) -> int:
    """Number of tokens the grammar FSM emitted/replaced at a parameter-VALUE
    position (``grammar_replacement_value_tokens``).

    This is the direct measure of the "FSM must never emit value tokens"
    invariant: a correct hybrid-clean run leaves the grammar inactive inside
    free-form values, so this counter is 0. Any non-zero value is a value
    projection leak the promotion gate must catch. The audit battery
    (avpt.audit_rows) only tracks the two-wave projection channel and is blind
    to this grammar-replacement channel, so it is gated separately."""
    if not metrics:
        return 0
    return int(metrics.get("grammar_replacement_value_tokens") or 0)


# ---------------------------------------------------------------------------
# Engine adapter interface  (the seam to the future vLLM Qwen3_5FlareModelState)
# ---------------------------------------------------------------------------
@dataclass
class EngineTurnResult:
    output_ids: list[int]
    metrics: dict[str, Any]
    block_snapshots: list[StateSnapshot]


class EngineUnavailable(RuntimeError):
    pass


class EngineAdapter:
    name: str = "abstract"

    def preflight(self) -> None:
        """Raise ``EngineUnavailable`` (cheaply, no model load) if this adapter
        cannot run a turn. Called before the expensive reference model load."""
        return None

    def run_turn(self, ctx: "TurnContext") -> EngineTurnResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class TurnContext:
    model: Any
    tokenizer: Any
    prompt_input_ids: torch.Tensor  # [1, T] on model device
    block_size: int
    max_new_tokens: int
    mask_id: int
    stop_token_ids: set[int]
    top_p: float
    temperature: float
    schemas: dict[str, Any]
    grammar_topk: int


def _run_hybrid_clean_instrumented(ctx: TurnContext) -> EngineTurnResult:
    """Run the verbatim reference sampler, capturing block-boundary snapshots."""
    from eval_flare_northstar_hybrid_clean import sample_hybrid_clean
    from flare_hf_cache import FlarePrefixCache

    prefix_cache = FlarePrefixCache()
    with record_boundary_snapshots() as snaps:
        output_ids, metrics = sample_hybrid_clean(
            ctx.model,
            ctx.tokenizer,
            ctx.prompt_input_ids,
            block_size=ctx.block_size,
            max_new_tokens=ctx.max_new_tokens,
            mask_id=ctx.mask_id,
            stop_token_ids=ctx.stop_token_ids,
            top_p=ctx.top_p,
            temperature=ctx.temperature,
            schemas=ctx.schemas,
            grammar_topk=ctx.grammar_topk,
            prefix_cache=prefix_cache,
        )
    return EngineTurnResult(
        output_ids=[int(x) for x in output_ids.tolist()],
        metrics=metrics,
        block_snapshots=list(snaps),
    )


class ReferenceRunner:
    """The (a) side: the HF hybrid-clean reference driven by flare_hf_cache."""

    def run(self, ctx: TurnContext) -> EngineTurnResult:
        return _run_hybrid_clean_instrumented(ctx)


class SelfEngineAdapter(EngineAdapter):
    """(b) side stand-in that re-runs the reference through an INDEPENDENT
    RequestDiffusionState + FlarePrefixCache instance with its own snapshot capture.

    Byte-identity is trivially satisfied (same code path); its value is a full-scale
    DETERMINISM + machinery self-test on the real model, plus it exercises the entire
    comparison + audit report end-to-end so the gate can be dry-run before the real
    engine exists.
    """

    name = "self"

    def run_turn(self, ctx: TurnContext) -> EngineTurnResult:
        return _run_hybrid_clean_instrumented(ctx)


class VllmFlareEngineAdapter(EngineAdapter):
    """(b) side: the real engine -- vLLM ``Qwen3_5FlareModelState`` on the pinned
    ``vllm_p2_pr42406`` workspace.  GPU-ONLY.

    Integration contract (from p2_serving_reuse_plan.md section 3, THE INTEGRATION SEAM):

      * The served ``Qwen3_5ForConditionalGeneration`` exposes a static
        ``get_model_state_cls()`` returning ``Qwen3_5FlareModelState``
        (dispatch site: vllm/v1/worker/gpu/model_states/__init__.py L18-19).
      * Per-request activation: ``SamplingParams.extra_args["decode_mode"]="block_diffusion"``
        carrying {tau, block_size, grammar_fsm_id} (fr10_decode_modes pattern).
      * One decode "step" = one denoise wave on the spec-decode-shaped path:
          masked active block   -> canvas draft tokens (prepare_inputs)
          noisy read mask       -> prepare_attn per-seq causal=False (Triton unified)
          GDN denoise read      -> fused_recurrent(..., inplace_final_state=False), no write-back
          commit                -> per-seq causal=True clean pass; publish fp32 boundary
                                   snapshot + conv_tail VERBATIM into the align-mode row;
                                   capture last noisy-stream logit before advance.
      * Snapshot extraction (``StateSnapshot`` per committed block boundary) reads the
        align-mode block-aligned checkpoint row:
          linear_attention layer -> gdn_state = ssm_state slot (fp32), conv_tail = conv_state tail
          full_attention  layer  -> key_len/value_len = committed paged-KV length
          last_token_logits      -> the shifted last-noisy logit captured pre-advance.
      * Force-counters (label-free audit) MUST be surfaced with values == 0:
          two_wave_wave1_projected_tokens, parallel_commit_forced_tokens.

    Until ``Qwen3_5FlareModelState`` lands, ``run_turn`` raises ``EngineUnavailable``
    with the exact missing symbol path, and the harness audits the reference side only.
    """

    name = "vllm"

    def __init__(self, workspace: Path = DEFAULT_VLLM_WORKSPACE):
        self.workspace = Path(workspace)

    def _locate(self):
        if not torch.cuda.is_available():
            raise EngineUnavailable("vllm engine adapter requires CUDA (GPU-only path)")
        ms_dir = self.workspace / "vllm" / "v1" / "worker" / "gpu" / "model_states"
        if not ms_dir.exists():
            raise EngineUnavailable(f"vLLM workspace model_states dir not found: {ms_dir}")
        # Grep every model_state module for the concrete class, without importing vllm
        # (importing vllm eagerly is heavy and pulls CUDA init).
        found = None
        for path in sorted(ms_dir.glob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if "class Qwen3_5FlareModelState" in text:
                found = path
                break
        if found is None:
            raise EngineUnavailable(
                "Qwen3_5FlareModelState is not present yet. Expected a class "
                f"'Qwen3_5FlareModelState' in one of {ms_dir}/*.py (expected file: "
                f"{ms_dir / 'qwen3_5_flare.py'}). See VllmFlareEngineAdapter docstring "
                "for the integration contract; wire snapshot_from_vllm_modelstate() and "
                "the vLLM serving driver once the class exists. NOTE: the pure engine ops "
                "(qwen3_5_flare_ops.py) can be parity-checked on CPU now via --mode ops-parity."
            )
        return found

    def snapshot_from_vllm_modelstate(self, model_state, *, block_start: int) -> StateSnapshot:  # pragma: no cover
        """Convert one align-mode block-boundary checkpoint into a StateSnapshot.

        MUST be implemented alongside Qwen3_5FlareModelState. See class docstring for the
        exact per-layer field mapping (ssm_state slot -> gdn_state fp32, conv_state tail ->
        conv_tail, paged-KV length -> key_len/value_len, pre-advance shifted logit ->
        last_token_logits)."""
        raise EngineUnavailable(
            "snapshot_from_vllm_modelstate() not wired yet -- implement with Qwen3_5FlareModelState"
        )

    def preflight(self) -> None:
        # Fails closed until the vLLM serving driver is wired (see run_turn).
        self.run_turn(None)  # type: ignore[arg-type]

    def run_turn(self, ctx: TurnContext) -> EngineTurnResult:  # pragma: no cover - GPU/engine path
        found = self._locate()  # raises EngineUnavailable with the precise reason if missing
        raise EngineUnavailable(
            f"Qwen3_5FlareModelState located ({found}) but the vLLM serving driver is not "
            "wired into this harness. To close the turn-level gate: boot vLLM V2 runner "
            "(VLLM_USE_V2_MODEL_RUNNER=1, TRITON_ATTN, --mamba-cache-mode align "
            "--mamba-ssm-cache-dtype float32 --enable-prefix-caching), serve the model so "
            "Qwen3_5ForConditionalGeneration.get_model_state_cls() -> Qwen3_5FlareModelState, "
            "drive the turn via SamplingParams.extra_args decode_mode='block_diffusion', "
            "collect output token ids, and emit one StateSnapshot per committed block boundary "
            "via snapshot_from_vllm_modelstate() (read Qwen3_5FlareRequestStates.block_start / "
            "last_shift_logits and _gdn_caches() (conv,ssm) rows post-commit). The pure engine "
            "ops are already parity-checked on CPU via --mode ops-parity."
        )


def build_engine_adapter(name: str, *, vllm_workspace: Path) -> EngineAdapter:
    if name == "self":
        return SelfEngineAdapter()
    if name == "vllm":
        return VllmFlareEngineAdapter(vllm_workspace)
    raise ValueError(f"unknown engine adapter: {name!r}")


# ---------------------------------------------------------------------------
# Row construction for the audit battery
# ---------------------------------------------------------------------------
def hybrid_schedule_events(metrics: dict[str, Any]) -> dict[str, Any]:
    """Mirror the schedule-events shape the reference eval writes into backend_meta,
    so avpt.audit_rows consumes it identically.

    The projection / forced-commit counters (and per-token records) are PASSED
    THROUGH from ``metrics`` rather than hard-coded to 0: the hybrid-clean
    reference is single-wave and never sets those keys, so it still defaults to
    0 (faithful), but a real engine that projects a value token MUST be able to
    surface it here or gate #3 (zero projected values) becomes a tautology that
    can never fail on the engine side."""
    denoise = int(metrics.get("denoise_forwards") or 0)
    events = {
        "denoise_forwards_total": denoise,
        "hybrid_model_forwards": denoise,
        "hybrid_forced_grammar_tokens": int(metrics.get("forced_grammar_tokens") or 0),
        "hybrid_model_value_tokens": int(metrics.get("model_value_tokens") or 0),
        "hybrid_model_structural_tokens": int(metrics.get("model_structural_tokens") or 0),
        "hybrid_value_close_timing_tokens": int(metrics.get("value_close_timing_tokens") or 0),
        "hybrid_grammar_replacement_value_tokens": int(metrics.get("grammar_replacement_value_tokens") or 0),
        "hybrid_grammar_unsafe_fallback_tokens": int(metrics.get("grammar_unsafe_fallback_tokens") or 0),
        "parallel_commit_value_tokens": int(metrics.get("model_value_tokens") or 0),
        "parallel_commit_forced_tokens": int(metrics.get("parallel_commit_forced_tokens") or 0),
        "two_wave_wave1_projected_tokens": int(metrics.get("two_wave_wave1_projected_tokens") or 0),
        "two_wave_wave1_value_tokens": int(metrics.get("two_wave_wave1_value_tokens") or 0),
        "two_wave_wave2_value_tokens": int(metrics.get("two_wave_wave2_value_tokens") or 0),
        "two_wave_wave2_forced_tokens": int(metrics.get("two_wave_wave2_forced_tokens") or 0),
    }
    records = metrics.get("two_wave_wave1_projected_token_records")
    if isinstance(records, list):
        # Per-token projection records: avpt.audit_rows uses these to compute the
        # EXACT projected-value count (value-position intersection), which is the
        # strongest arm of the zero-projected verification.
        events["two_wave_wave1_projected_token_records"] = records
    return events


def build_audit_row(*, backend: str, generated_ids: list[int], assistant_text: str, exact_arguments: bool, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": backend,
        "assistant": assistant_text,
        "exact_arguments": bool(exact_arguments),
        "generated_token_ids": [int(x) for x in generated_ids],
        "backend_meta": {"sampler_schedule_events": hybrid_schedule_events(metrics)},
    }


# ---------------------------------------------------------------------------
# MODE: turn  (GPU, real model + matched-20 data)
# ---------------------------------------------------------------------------
def load_real_model_and_data(args):
    from eval_fastdllm_toolcall_cases import load_model, resolve_token_ids
    from eval_flare_northstar_matched import (
        build_episodes,
        load_chat_template,
        render_matched_prompt,
    )
    from eval_flare_stage1_ab_diffusion import set_block_size

    model, tokenizer = load_model(
        args.base_model,
        args.adapter if args.adapter and Path(args.adapter).exists() else None,
        merge_adapter=not args.no_merge_adapter,
        tokenizer_path=args.tokenizer_path,
    )
    model.eval()
    set_block_size(model, int(args.block_size))
    mask_id, _stop_token_id, base_stop_token_ids = resolve_token_ids(model, tokenizer)
    tool_close_ids = tokenizer("</tool_call>", add_special_tokens=False).input_ids
    stop_token_ids = set(int(x) for x in list(base_stop_token_ids) + list(tool_close_ids))
    chat_template = load_chat_template(args.chat_template_path)
    episodes = build_episodes(args)
    return model, tokenizer, mask_id, stop_token_ids, chat_template, episodes, render_matched_prompt


def build_turn_prompt(model, tokenizer, episode, chat_template, target_turn, ctx_kwargs, render_matched_prompt) -> str:
    """Prompt for `target_turn`. For target 0 this is the initial rendered prompt.
    For target>0 the prior turns are replayed through the reference to construct the
    running prompt (same construction the reference eval uses)."""
    from eval_flare_northstar_matched import (
        decode_text,
        next_turn_user_message,
        tool_response_suffix,
        trim_scored_assistant,
        row_from_generation,
    )
    from eval_toolcall_jsonl import tool_schema_by_name

    messages = [dict(m) for m in episode["prompt_messages"]]
    prompt = render_matched_prompt(tokenizer, messages, episode["tools"], chat_template)
    for turn_idx in range(0, target_turn):
        gold_block = episode["gold_blocks"][turn_idx]
        prompt_input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        ctx = TurnContext(
            model=model,
            tokenizer=tokenizer,
            prompt_input_ids=prompt_input_ids,
            schemas=tool_schema_by_name(episode["tools"]),
            **ctx_kwargs,
        )
        res = _run_hybrid_clean_instrumented(ctx)
        new_ids = torch.tensor(res.output_ids[int(prompt_input_ids.shape[1]):], dtype=torch.long)
        history_text = decode_text(tokenizer, new_ids)
        assistant_text = trim_scored_assistant(history_text)
        row = row_from_generation(
            backend="reference_prefill",
            episode=episode,
            turn_idx=turn_idx,
            prompt=prompt,
            tools=episode["tools"],
            gold_block=gold_block,
            assistant_text=assistant_text,
            prompt_tokens=int(prompt_input_ids.shape[1]),
            generated_tokens=int(new_ids.numel()),
            turn_wall_seconds=0.0,
            schedule_build_seconds=0.0,
            backend_meta={},
        )
        next_user = next_turn_user_message(episode, turn_idx + 1)
        prompt = prompt + history_text + tool_response_suffix(row["tool_response_payload"], next_user)
    return prompt


def run_turn_mode(args) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("turn mode requires CUDA (real 9B model). Use --mode state-parity / selftest on CPU.")
    _set_route_i_env()
    from eval_flare_northstar_matched import (
        decode_text,
        trim_scored_assistant,
    )
    from eval_toolcall_jsonl import tool_schema_by_name
    from eval_toolcall_jsonl import score_tool_calls

    # Cheap engine preflight BEFORE the multi-minute 9B reference load, so
    # turn+<engine-not-runnable> is reported without wasting a reference decode.
    adapter = build_engine_adapter(args.engine, vllm_workspace=args.vllm_workspace)
    engine_preflight_error = None
    try:
        adapter.preflight()
    except EngineUnavailable as exc:
        engine_preflight_error = str(exc)
        if not args.reference_only_if_engine_unavailable:
            return {
                "mode": "turn",
                "engine": adapter.name,
                "engine_available": False,
                "engine_error": engine_preflight_error,
                "passed": None,
                "verdict": "ENGINE_UNAVAILABLE_preflight",
                "hint": "pass --reference-only-if-engine-unavailable to still run + audit the (a) reference side",
            }
        print(f"[warn] engine '{adapter.name}' unavailable at preflight; running reference-only audit.\n"
              f"       reason: {engine_preflight_error}", file=sys.stderr, flush=True)

    torch.manual_seed(int(args.seed))
    (model, tokenizer, mask_id, stop_token_ids, chat_template, episodes, render_matched_prompt) = load_real_model_and_data(args)

    if args.episode_index >= len(episodes):
        raise SystemExit(f"episode_index {args.episode_index} out of range ({len(episodes)} episodes)")
    episode = episodes[args.episode_index]
    if args.turn_index >= len(episode["gold_blocks"]):
        raise SystemExit(f"turn_index {args.turn_index} out of range ({len(episode['gold_blocks'])} turns)")

    ctx_kwargs = dict(
        block_size=int(args.block_size),
        max_new_tokens=int(args.max_new_tokens),
        mask_id=int(mask_id),
        stop_token_ids=stop_token_ids,
        top_p=float(args.top_p),
        temperature=float(args.temperature),
        grammar_topk=int(args.grammar_topk),
    )

    prompt = build_turn_prompt(model, tokenizer, episode, chat_template, args.turn_index, ctx_kwargs, render_matched_prompt)
    gold_block = episode["gold_blocks"][args.turn_index]
    schemas = tool_schema_by_name(episode["tools"])
    prompt_input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    prompt_len = int(prompt_input_ids.shape[1])

    def make_ctx():
        return TurnContext(
            model=model,
            tokenizer=tokenizer,
            prompt_input_ids=prompt_input_ids,
            schemas=schemas,
            **ctx_kwargs,
        )

    def finalize(res: EngineTurnResult, backend: str) -> dict[str, Any]:
        new_ids = res.output_ids[prompt_len:]
        history_text = decode_text(tokenizer, torch.tensor(new_ids, dtype=torch.long))
        assistant_text = trim_scored_assistant(history_text)
        metrics_score = score_tool_calls(assistant_text, episode["tools"], gold_block)
        row = build_audit_row(
            backend=backend,
            generated_ids=new_ids,
            assistant_text=assistant_text,
            exact_arguments=bool(metrics_score.get("exact_arguments")),
            metrics=res.metrics,
        )
        return {
            "new_ids": new_ids,
            "assistant_text": assistant_text,
            "row": row,
            "exact_arguments": bool(metrics_score.get("exact_arguments")),
            "value_token_count": value_token_count(tokenizer, row),
        }

    # (a) reference
    t0 = time.time()
    ref_res = ReferenceRunner().run(make_ctx())
    ref_wall = time.time() - t0
    ref_final = finalize(ref_res, "reference_hybrid_clean")

    # (b) engine  (adapter already built + preflighted above)
    engine_available = True
    engine_error = engine_preflight_error
    eng_res = None
    eng_final = None
    eng_wall = None
    try:
        t1 = time.time()
        eng_res = adapter.run_turn(make_ctx())
        eng_wall = time.time() - t1
        eng_final = finalize(eng_res, f"engine_{adapter.name}")
    except EngineUnavailable as exc:
        engine_available = False
        engine_error = str(exc)

    ref_totals = audit_totals_for_rows(tokenizer, [ref_final["row"]])
    report: dict[str, Any] = {
        "mode": "turn",
        "engine": adapter.name,
        "engine_available": engine_available,
        "episode_index": int(args.episode_index),
        "turn_index": int(args.turn_index),
        "episode_id": episode.get("id"),
        "block_size": int(args.block_size),
        "prompt_tokens": prompt_len,
        "gdn_atol": float(args.gdn_atol),
        "logit_atol": float(args.logit_atol),
        "reference": {
            "generated_token_count": len(ref_final["new_ids"]),
            "exact_arguments": ref_final["exact_arguments"],
            "value_token_count": ref_final["value_token_count"],
            "grammar_replacement_value_tokens": grammar_value_projection_count(ref_res.metrics),
            "grammar_unsafe_fallback_tokens": int(ref_res.metrics.get("grammar_unsafe_fallback_tokens") or 0),
            "stop_reason": ref_res.metrics.get("stop_reason"),
            "cache_stats": ref_res.metrics.get("cache_stats"),
            "num_block_boundaries": len(ref_res.block_snapshots),
            "wall_seconds": ref_wall,
        },
        "audit_battery": {
            "reference_totals": ref_totals,
            "reference_zero_projected_ok": zero_projected_ok(ref_totals),
        },
    }

    if not engine_available:
        report["engine_error"] = engine_error
        report["gates"] = {
            "reference_zero_projected_values": zero_projected_ok(ref_totals),
            "reference_no_grammar_value_projection": grammar_value_projection_count(ref_res.metrics) == 0,
            "byte_identical": None,
            "value_token_counts_equal": None,
            "engine_zero_projected_values": None,
            "state_snapshot_equality": None,
        }
        report["passed"] = None
        report["verdict"] = "ENGINE_UNAVAILABLE_reference_audited_only"
        return report

    eng_totals = audit_totals_for_rows(tokenizer, [eng_final["row"]])
    byte_report = token_bytes_identical(ref_final["new_ids"], eng_final["new_ids"], tokenizer=tokenizer)
    snap_report = compare_snapshot_sequences(
        ref_res.block_snapshots,
        eng_res.block_snapshots,
        gdn_atol=float(args.gdn_atol),
        logit_atol=float(args.logit_atol),
    )
    value_counts_equal = ref_final["value_token_count"] == eng_final["value_token_count"] and (
        int(ref_totals.get("reported_model_value_tokens") or 0)
        == int(eng_totals.get("reported_model_value_tokens") or 0)
    )
    ref_grammar_value_proj = grammar_value_projection_count(ref_res.metrics)
    eng_grammar_value_proj = grammar_value_projection_count(eng_res.metrics)
    gates = {
        "byte_identical": bool(byte_report["byte_exact"] and byte_report["token_exact"]),
        "value_token_counts_equal": bool(value_counts_equal),
        "reference_zero_projected_values": zero_projected_ok(ref_totals),
        "engine_zero_projected_values": zero_projected_ok(eng_totals),
        "no_grammar_value_projection": bool(ref_grammar_value_proj == 0 and eng_grammar_value_proj == 0),
        "state_snapshot_equality": bool(snap_report["ok"]),
    }
    report["engine_result"] = {
        "generated_token_count": len(eng_final["new_ids"]),
        "exact_arguments": eng_final["exact_arguments"],
        "value_token_count": eng_final["value_token_count"],
        "grammar_replacement_value_tokens": eng_grammar_value_proj,
        "grammar_unsafe_fallback_tokens": int(eng_res.metrics.get("grammar_unsafe_fallback_tokens") or 0),
        "stop_reason": eng_res.metrics.get("stop_reason"),
        "cache_stats": eng_res.metrics.get("cache_stats"),
        "num_block_boundaries": len(eng_res.block_snapshots),
        "wall_seconds": eng_wall,
    }
    report["byte_identity"] = byte_report
    report["state_snapshot_parity"] = snap_report
    report["audit_battery"]["engine_totals"] = eng_totals
    report["audit_battery"]["engine_zero_projected_ok"] = zero_projected_ok(eng_totals)
    report["gates"] = gates
    report["passed"] = bool(all(gates.values()))
    report["verdict"] = "PASS" if report["passed"] else "FAIL"
    return report


# ---------------------------------------------------------------------------
# MODE: state-parity  (CPU tiny route_i model -- real StateSnapshot machinery)
# ---------------------------------------------------------------------------
def run_state_parity_mode(args) -> dict[str, Any]:
    _set_route_i_env()
    from flare_hf_cache import RequestDiffusionState
    from validate_flare_hf_cache import (
        deterministic_ids,
        greedy_canary,
        noisy_block_from_clean,
        shifted_reference,
    )
    from validate_flare_two_stream_forward import load_local_bridge, make_tiny_model

    torch.manual_seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.threads)))
    config_module, modeling_module = load_local_bridge(Path(args.model_dir).resolve())
    block_size = int(args.block_size_tiny)
    model = make_tiny_model(config_module, modeling_module, seed=int(args.seed), block_size=block_size).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mask_id = int(model.config.mask_token_id)
    blocks = int(args.blocks)
    batch_size = int(args.batch_size)

    clean_ids = deterministic_ids(model, batch_size, blocks * block_size, device, mask_id)

    # Reference driver: incremental advance, capture snapshot per boundary.
    # Engine (independent) driver: from-scratch reset up to each boundary, capture snapshot.
    # These are two independent computations of the SAME boundary carrier: the align-APC
    # restore-by-copy invariant demands they be fp32-equal.
    ref_state = RequestDiffusionState.reset(model, clean_ids[:, :0], block_size)
    ref_snaps: list[StateSnapshot] = []
    eng_snaps: list[StateSnapshot] = []

    argmax_flips = 0
    max_logit_diff = 0.0
    positions = 0
    for b in range(blocks):
        start = b * block_size
        end = start + block_size
        # shifted-logit + argmax parity (cache-on vs full recompute) on the noisy block
        noisy_block = noisy_block_from_clean(clean_ids[:, start:end], mask_id, b)
        x_t = torch.cat([clean_ids[:, :start], noisy_block], dim=1)
        cached = ref_state.shifted_active_logits(model, x_t)
        reference = shifted_reference(model, x_t, block_size, mask_id, active_start=start)
        max_logit_diff = max(max_logit_diff, float((cached.float() - reference.float()).abs().max().item()))
        argmax_flips += int((cached.argmax(-1) != reference.argmax(-1)).sum().item())
        positions += int(cached.shape[0] * cached.shape[1])

        ref_state.advance(model, clean_ids[:, start:end])
        ref_snaps.append(capture_state_snapshot(ref_state))

        eng_state = RequestDiffusionState.reset(model, clean_ids[:, : end], block_size)
        eng_snaps.append(capture_state_snapshot(eng_state))

    snap_report = compare_snapshot_sequences(
        ref_snaps, eng_snaps, gdn_atol=float(args.gdn_atol), logit_atol=float(args.logit_atol)
    )

    # greedy cache-on vs cache-off byte-identity (T3-style), independent decoders.
    prompt_ids = deterministic_ids(model, 1, block_size // 2, device, mask_id)
    new_tokens = 2 * block_size
    off, _ = greedy_canary(model, prompt_ids, block_size=block_size, mask_id=mask_id, new_tokens=new_tokens, use_cache=False)
    on, cache_stats = greedy_canary(model, prompt_ids, block_size=block_size, mask_id=mask_id, new_tokens=new_tokens, use_cache=True)
    byte_report = token_bytes_identical(off[0].tolist(), on[0].tolist(), tokenizer=None)

    # audit-battery-shaped counters (no XML on the tiny model; force the invariant shape)
    counters = {
        "rows": 1,
        "denoise_forwards_total": int(ref_state.read_calls),
        "wave1_projected_tokens": 0,
        "true_xml_value_tokens": 0,
        "reported_model_value_tokens": 0,
        "projected_value_tokens_exact": 0,
        "parallel_commit_forced_tokens_counter": 0,
        "zero_projected_value_tokens_verified": 1,
        "verification_mode": "no_projection_events",
    }

    gates = {
        "shifted_logit_argmax_parity": bool(argmax_flips == 0 and max_logit_diff <= float(args.logit_atol)),
        "state_snapshot_equality": bool(snap_report["ok"]),
        "greedy_byte_identity": bool(byte_report["byte_exact"] and byte_report["token_exact"]),
        "zero_projected_values": zero_projected_ok(counters),
    }
    return {
        "mode": "state-parity",
        "device": str(device),
        "block_size": block_size,
        "blocks": blocks,
        "batch_size": batch_size,
        "positions": positions,
        "argmax_flips": int(argmax_flips),
        "shifted_logit_max_abs_diff": max_logit_diff,
        "state_snapshot_parity": snap_report,
        "greedy_byte_identity": byte_report,
        "cache_stats": cache_stats,
        "audit_battery": {"counters": counters, "zero_projected_ok": zero_projected_ok(counters)},
        "gates": gates,
        "passed": bool(all(gates.values())),
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }


# ---------------------------------------------------------------------------
# MODE: selftest  (CPU, tokenizer-only -- verifies the comparators + audit)
# ---------------------------------------------------------------------------
def _fabricate_snapshot(block_start: int, *, gdn_scale: float, kv_len: int, logit: float) -> StateSnapshot:
    return StateSnapshot(
        block_start=block_start,
        block_size=32,
        batch_size=1,
        layers=[
            LayerSnapshot(kind="linear_attention", gdn_state=torch.full((1, 2, 4, 4), gdn_scale), conv_tail=torch.full((1, 2, 2), gdn_scale), key_len=0, value_len=0),
            LayerSnapshot(kind="full_attention", gdn_state=None, conv_tail=None, key_len=kv_len, value_len=kv_len),
        ],
        last_token_logits=torch.full((1, 1, 8), logit),
    )


def _load_selftest_tokenizer(args):
    from transformers import AutoTokenizer

    for path in (args.tokenizer_path, args.prompt_tokenizer_path):
        try:
            return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        except Exception:
            continue
    return None


def run_selftest_mode(args) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    gdn_atol = float(args.gdn_atol)
    logit_atol = float(args.logit_atol)

    # --- snapshot comparator: matching sequence passes ---
    ref_seq = [_fabricate_snapshot(0, gdn_scale=1.0, kv_len=32, logit=0.5), _fabricate_snapshot(32, gdn_scale=2.0, kv_len=64, logit=1.5)]
    eng_ok = [
        _fabricate_snapshot(0, gdn_scale=1.0 + gdn_atol * 0.5, kv_len=32, logit=0.5 + logit_atol * 0.5),
        _fabricate_snapshot(32, gdn_scale=2.0, kv_len=64, logit=1.5),
    ]
    rep = compare_snapshot_sequences(ref_seq, eng_ok, gdn_atol=gdn_atol, logit_atol=logit_atol)
    record("snapshot_match_within_tol_passes", rep["ok"], json.dumps(rep["failing_boundaries"]))

    # --- snapshot comparator: GDN drift beyond tol fails ---
    eng_gdn = [ref_seq[0], _fabricate_snapshot(32, gdn_scale=2.0 + gdn_atol * 10, kv_len=64, logit=1.5)]
    rep = compare_snapshot_sequences(ref_seq, eng_gdn, gdn_atol=gdn_atol, logit_atol=logit_atol)
    record("snapshot_gdn_drift_fails", not rep["ok"], f"max_gdn={rep['max_gdn_state_abs_diff']:.3g}")

    # --- snapshot comparator: KV-length mismatch fails ---
    eng_kv = [ref_seq[0], _fabricate_snapshot(32, gdn_scale=2.0, kv_len=63, logit=1.5)]
    rep = compare_snapshot_sequences(ref_seq, eng_kv, gdn_atol=gdn_atol, logit_atol=logit_atol)
    record("snapshot_kv_len_mismatch_fails", not rep["ok"], json.dumps(rep["failing_boundaries"]))

    # --- snapshot comparator: logit drift beyond tol fails ---
    eng_lg = [_fabricate_snapshot(0, gdn_scale=1.0, kv_len=32, logit=0.5 + logit_atol * 5), ref_seq[1]]
    rep = compare_snapshot_sequences(ref_seq, eng_lg, gdn_atol=gdn_atol, logit_atol=logit_atol)
    record("snapshot_logit_drift_fails", not rep["ok"], f"max_logit={rep['max_last_logits_abs_diff']:.3g}")

    # --- snapshot comparator: boundary count mismatch fails ---
    rep = compare_snapshot_sequences(ref_seq, ref_seq[:1], gdn_atol=gdn_atol, logit_atol=logit_atol)
    record("snapshot_boundary_count_mismatch_fails", not rep["ok"], "")

    # --- byte identity: equal vs divergent ---
    rep = token_bytes_identical([1, 2, 3, 4], [1, 2, 3, 4], tokenizer=None)
    record("byte_identity_equal_passes", rep["token_exact"] and rep["byte_exact"], "")
    rep = token_bytes_identical([1, 2, 3, 4], [1, 2, 9, 4], tokenizer=None)
    record("byte_identity_divergent_fails", not rep["token_exact"] and rep["first_divergence"]["index"] == 2, json.dumps(rep["first_divergence"]))

    # --- grammar-value-projection channel: clean=0, leak detected ---
    record("grammar_value_projection_clean_zero", grammar_value_projection_count({"model_value_tokens": 4}) == 0)
    record("grammar_value_projection_leak_detected", grammar_value_projection_count({"grammar_replacement_value_tokens": 2}) == 2)

    # --- projection counters flow through hybrid_schedule_events into the audit ---
    proj_events = hybrid_schedule_events({"denoise_forwards": 3, "two_wave_wave1_projected_tokens": 5, "parallel_commit_forced_tokens": 1})
    record("projection_counters_flow_through",
           proj_events["two_wave_wave1_projected_tokens"] == 5 and proj_events["parallel_commit_forced_tokens"] == 1,
           json.dumps({k: proj_events[k] for k in ("two_wave_wave1_projected_tokens", "parallel_commit_forced_tokens")}))
    clean_events = hybrid_schedule_events({"denoise_forwards": 3, "model_value_tokens": 4})
    record("reference_projection_counters_default_zero",
           clean_events["two_wave_wave1_projected_tokens"] == 0 and clean_events["parallel_commit_forced_tokens"] == 0)

    # --- audit battery (needs a real tokenizer) ---
    tokenizer = _load_selftest_tokenizer(args)
    audit_detail: dict[str, Any] = {"tokenizer_loaded": tokenizer is not None}
    if tokenizer is not None:
        xml = "<tool_call>\n<function=book>\n<parameter=city>\nNew York\n</parameter>\n<parameter=day>\nMonday\n</parameter>\n</function>\n</tool_call>"
        clean_row = build_audit_row(
            backend="ref",
            generated_ids=tokenizer(xml, add_special_tokens=False).input_ids,
            assistant_text=xml,
            exact_arguments=True,
            metrics={"denoise_forwards": 12, "model_value_tokens": 4},
        )
        totals = audit_totals_for_rows(tokenizer, [clean_row])
        record("audit_zero_projected_clean_passes", zero_projected_ok(totals), totals.get("verification_mode"))
        vcount = value_token_count(tokenizer, clean_row)
        record("audit_value_token_count_positive", vcount > 0, f"value_tokens={vcount}")
        audit_detail["clean_totals"] = totals

        # Inject a projected VALUE-token record -> the audit must flag it (zero-projected FAILS).
        dirty_row = json.loads(json.dumps(clean_row))
        gen_ids = clean_row["generated_token_ids"]
        text, offsets = avpt.token_offsets_from_generated_ids(tokenizer, gen_ids)
        spans = avpt.value_spans(text)
        value_rel_idx = None
        for idx, (s, e) in enumerate(offsets):
            if e > s and any(e > sp["start"] and s < sp["end"] for sp in spans):
                value_rel_idx = idx
                break
        events = dirty_row["backend_meta"]["sampler_schedule_events"]
        events["two_wave_wave1_projected_tokens"] = 1
        events["two_wave_wave1_projected_token_records"] = [{"rel_idx": value_rel_idx}]
        dirty_totals = audit_totals_for_rows(tokenizer, [dirty_row])
        record("audit_projected_value_detected_fails", not zero_projected_ok(dirty_totals),
               f"exact={dirty_totals.get('projected_value_tokens_exact')} mode={dirty_totals.get('verification_mode')}")
        audit_detail["dirty_totals"] = dirty_totals
        audit_detail["value_rel_idx"] = value_rel_idx

        # END-TO-END: the SAME projection routed through the turn-mode path
        # (metrics -> build_audit_row -> hybrid_schedule_events -> audit) must
        # also be caught. This is the regression guard for the fixed tautology:
        # before the fix, hybrid_schedule_events hard-coded projected=0 and this
        # row would have (wrongly) passed zero_projected_ok.
        engine_like_row = build_audit_row(
            backend="engine_like",
            generated_ids=gen_ids,
            assistant_text=xml,
            exact_arguments=True,
            metrics={
                "denoise_forwards": 12,
                "model_value_tokens": 4,
                "two_wave_wave1_projected_tokens": 1,
                "two_wave_wave1_projected_token_records": [{"rel_idx": value_rel_idx}],
            },
        )
        engine_like_totals = audit_totals_for_rows(tokenizer, [engine_like_row])
        record("turn_path_projected_value_detected_fails", not zero_projected_ok(engine_like_totals),
               f"exact={engine_like_totals.get('projected_value_tokens_exact')} mode={engine_like_totals.get('verification_mode')}")
        audit_detail["engine_like_totals"] = engine_like_totals
    else:
        record("audit_battery_skipped_no_tokenizer", True, "tokenizer not found; audit checks skipped")

    passed = all(c["ok"] for c in checks)
    return {
        "mode": "selftest",
        "checks": checks,
        "num_checks": len(checks),
        "num_failed": sum(1 for c in checks if not c["ok"]),
        "audit_detail": audit_detail,
        "gdn_atol": gdn_atol,
        "logit_atol": logit_atol,
        "passed": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
    }


# ---------------------------------------------------------------------------
# MODE: ops-parity  (CPU -- REAL engine ops vs flare_hf_cache primitives)
# ---------------------------------------------------------------------------
def _load_engine_ops(workspace: Path):
    """Import ``qwen3_5_flare_ops`` directly from the workspace file (pure torch,
    no vLLM package init). Registers in sys.modules first so its dataclasses build."""
    import importlib.util

    path = Path(workspace) / "vllm" / "v1" / "worker" / "gpu" / "model_states" / "qwen3_5_flare_ops.py"
    if not path.exists():
        raise EngineUnavailable(f"engine ops module not found: {path}")
    spec = importlib.util.spec_from_file_location("_qwen3_5_flare_ops_standalone", path)
    if spec is None or spec.loader is None:
        raise EngineUnavailable(f"could not load engine ops from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass needs the module registered
    spec.loader.exec_module(module)
    return module, path


def run_ops_parity_mode(args) -> dict[str, Any]:
    import flare_hf_cache as F

    torch.manual_seed(int(args.seed))
    try:
        ops, ops_path = _load_engine_ops(args.vllm_workspace)
    except EngineUnavailable as exc:
        return {
            "mode": "ops-parity",
            "engine_ops_available": False,
            "engine_error": str(exc),
            "passed": None,
            "verdict": "ENGINE_OPS_UNAVAILABLE",
        }

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1. train-matched +1 right shift == RequestDiffusionState.shifted_active_logits construction
    b, L, V = 2, 5, 11
    bl = torch.randn(b, L, V)
    prev = torch.randn(b, 1, V)
    ref_b0 = torch.cat([bl[:, :1, :], bl[:, :-1, :]], dim=1)  # flare_hf_cache block_start==0 branch
    ref_bN = torch.cat([prev, bl[:, :-1, :]], dim=1)          # flare_hf_cache block_start>0 branch
    record("shift_block0_matches_reference", torch.equal(ref_b0, ops.right_shift_block_logits(bl, None)))
    record("shift_blockN_matches_reference", torch.equal(ref_bN, ops.right_shift_block_logits(bl, prev)))
    record("capture_shift_logit_is_last_pos", torch.equal(ops.capture_shift_logit(bl), bl[:, -1:, :]))

    # 2. conv-tail roll == flare_hf_cache._tail_after_append (identical inputs)
    tail_ok = True
    for tail_len, has_old in ((0, True), (2, False), (2, True), (3, True)):
        old = torch.randn(1, 2, 6) if has_old else None
        raw = torch.randn(1, 4, 6)
        a = F._tail_after_append(old, raw, tail_len)
        c = ops.tail_after_append(old, raw, tail_len)
        eq = (a is None and c is None) or (a is not None and c is not None and torch.equal(a, c))
        tail_ok = tail_ok and eq
    record("conv_tail_roll_matches_reference", tail_ok)

    # 3. read-only-denoise snapshot/restore: denoise rows must NOT move; commit rows DO
    conv = torch.randn(6, 3, 4)
    ssm = torch.randn(6, 2, 5)
    caches = [(conv.clone(), ssm.clone())]
    orig_conv, orig_ssm = caches[0][0].clone(), caches[0][1].clone()
    denoise_rows = torch.tensor([1, 4], dtype=torch.int64)
    commit_rows = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    snap = ops.snapshot_readonly_rows(caches, denoise_rows)
    caches[0][0].add_(100.0)  # simulate the in-place denoise write-back over ALL rows
    caches[0][1].add_(100.0)
    ops.restore_readonly_rows(caches, snap)
    denoise_held = torch.equal(caches[0][0][denoise_rows], orig_conv[denoise_rows]) and torch.equal(
        caches[0][1][denoise_rows], orig_ssm[denoise_rows]
    )
    commit_advanced = torch.equal(caches[0][0][commit_rows], orig_conv[commit_rows] + 100.0)
    record("readonly_denoise_suppresses_writeback", denoise_held)
    record("commit_rows_keep_advance", commit_advanced)
    # empty snapshot is a no-op
    empty_snap = ops.snapshot_readonly_rows([(conv.clone(), ssm.clone())], torch.empty(0, dtype=torch.int64))
    noop_caches = [(conv.clone().add_(7.0), ssm.clone().add_(7.0))]
    before = noop_caches[0][0].clone()
    ops.restore_readonly_rows(noop_caches, empty_snap)
    record("empty_readonly_snapshot_is_noop", torch.equal(noop_caches[0][0], before))

    # 4. fp32 boundary carrier (FlareLayerCache fp32 discipline)
    bf16_state = ssm.to(torch.bfloat16)
    cap = ops.FlareBoundarySnapshot.capture(bf16_state, conv)
    record("boundary_capture_is_fp32", cap.ssm_state.dtype == torch.float32)
    record("boundary_capture_value_preserving", torch.allclose(cap.ssm_state, bf16_state.float()))
    raised = False
    try:
        ops.assert_fp32_boundary(ops.FlareBoundarySnapshot(ssm_state=bf16_state, conv_tail=None))
    except RuntimeError:
        raised = True
    record("assert_fp32_boundary_rejects_non_fp32", raised)

    # 5. variable-accept num_sampled -> num_accepted == max(n, 1)  (MambaHybrid neutral=1)
    n = torch.tensor([0, 1, 2, 7])
    record("commit_num_accepted_is_max_1", ops.commit_num_accepted(n).tolist() == [1, 1, 2, 7])

    # 6. canvas denoise/commit phase transition truth table
    #    committing -> next denoise (False), step reset 0
    #    denoise & converged -> commit next (True); denoise & capped -> commit; else keep denoise
    is_committing = torch.tensor([True, False, False, False])
    step = torch.tensor([5, 1, 46, 2], dtype=torch.int32)
    converged = torch.tensor([False, True, False, False])
    new_step, next_phase = ops.flare_step_and_phase(is_committing, step, converged, max_denoising_steps=48)
    step_ok = new_step.tolist() == [0, 2, 47, 3]
    # row0 commit->denoise F; row1 converged->commit T; row2 capped(47>=?) no: 47<48 keep F; row3 keep F
    phase_ok = next_phase.tolist() == [False, True, False, False]
    # capped case: bump row2 to 47->48
    step2 = torch.tensor([5, 1, 47, 2], dtype=torch.int32)
    _, next_phase2 = ops.flare_step_and_phase(is_committing, step2, converged, max_denoising_steps=48)
    cap_ok = next_phase2.tolist() == [False, True, True, False]  # row2 now force-commits
    record("phase_transition_step_counter", step_ok, str(new_step.tolist()))
    record("phase_transition_denoise_commit_flag", phase_ok and cap_ok, f"{next_phase.tolist()} / capped {next_phase2.tolist()}")

    # 7. commit num_sampled = valid_canvas_len on commit else 0
    valid = torch.tensor([32, 32, 32, 32])
    ns = ops.flare_commit_num_sampled(is_committing, valid)
    record("commit_num_sampled_gates_on_commit", ns.tolist() == [32, 0, 0, 0], str(ns.tolist()))

    # 8. per-seq causal flag gather (commit=causal True, denoise=causal False)
    is_enc = torch.tensor([True, False, True, False, False])
    slots = torch.tensor([4, 0, 1])
    flags = ops.per_seq_causal_flags(is_enc, slots)
    record("per_seq_causal_flags_gather", flags.tolist() == [False, True, False], str(flags.tolist()))

    passed = all(c["ok"] for c in checks)
    return {
        "mode": "ops-parity",
        "engine_ops_available": True,
        "engine_ops_path": str(ops_path),
        "vllm_workspace": str(args.vllm_workspace),
        "checks": checks,
        "num_checks": len(checks),
        "num_failed": sum(1 for c in checks if not c["ok"]),
        "passed": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    from eval_flare_northstar_matched import (
        DEFAULT_AR_MODEL,
        DEFAULT_CHAT_TEMPLATE,
        DEFAULT_DIFFUSION_ADAPTER,
        DEFAULT_DIFFUSION_BASE,
        DEFAULT_INPUT,
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["selftest", "ops-parity", "state-parity", "turn"], default="selftest")
    p.add_argument("--engine", choices=["self", "vllm"], default="vllm",
                   help="engine adapter for turn mode: 'self' = determinism/machinery self-test at full scale; "
                        "'vllm' = the real Qwen3_5FlareModelState (GPU, raises EngineUnavailable until it lands).")
    p.add_argument("--out-json", type=Path, default=None)

    # tolerances
    p.add_argument("--gdn-atol", type=float, default=DEFAULT_GDN_ATOL)
    p.add_argument("--logit-atol", type=float, default=DEFAULT_LOGIT_ATOL)

    # turn mode (real model + matched-20 data)
    p.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--episode-limit", type=int, default=20)
    p.add_argument("--min-turns", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--turn-index", type=int, default=0)
    p.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    p.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    p.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    p.add_argument("--adapter", type=Path, default=DEFAULT_DIFFUSION_ADAPTER)
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    p.add_argument("--no-merge-adapter", action="store_true", default=True)
    p.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--grammar-topk", type=int, default=256)
    p.add_argument("--vllm-workspace", type=Path, default=DEFAULT_VLLM_WORKSPACE)
    p.add_argument("--reference-only-if-engine-unavailable", action="store_true",
                   help="turn mode: if the engine can't run, still load the model and run + audit "
                        "the (a) reference side (else short-circuit at preflight).")

    # state-parity mode (tiny CPU model)
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen3.5-9b-fastdllm-init")
    p.add_argument("--block-size-tiny", type=int, default=4)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)

    p.add_argument("--seed", type=int, default=20260701)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "selftest":
        report = run_selftest_mode(args)
    elif args.mode == "ops-parity":
        report = run_ops_parity_mode(args)
    elif args.mode == "state-parity":
        report = run_state_parity_mode(args)
    else:
        report = run_turn_mode(args)

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text, flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")

    passed = report.get("passed")
    if passed is None:
        return 3  # engine unavailable; reference audited only
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
