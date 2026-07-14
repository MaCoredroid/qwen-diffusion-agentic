#!/usr/bin/env python
# coding=utf-8
"""X.1(a) — READ_WINDOW_ARG read-grounding-weighted L_diff (training hook).

SPEC: k_raise_campaign_design.md SECTION X.1(a) (P0 probe). This is the NEW loss-config
module that carries the ENTIRE X.1(a) objective. It is imported by an ENV-GATED, strict
no-op augmentation in the model's `_argument_span_loss_weights` (modeling.py), activated
ONLY when FASTDLLM_READ_WINDOW_ARG_LOSS_WEIGHT != 1.0. The plain conversion path (arm A /
twinK1) does NOT set that env, so its loss-weight code path and byte-reproducibility are
UNCHANGED and it never enters any X.1 code. Same discipline as scripts/v1_copy_span_infill.py.

WHAT X.1(a) PRESCRIBES (implemented here, exactly):
  * Add a READ_WINDOW_ARG span class = the VALUE token positions of `read_file` window
    arguments (schema key in {limit, offset}) inside <parameter=NAME> ... </parameter>.
    Up-weight its K=1 sequential CE (extends VALUE_SPAN_LOSS_WEIGHT=2.0 -> 4..6 on this
    class only). It is DERIVED (never joint-committed) so it stays sequential; this is a
    per-token CE weight, not a masking change.
  * The tagger reuses the census/marker-subsequence detector, RESTRICTED to arg bodies whose
    schema key is limit/offset. The discriminating token subsequences (Qwen3.5 tokenizer):
        <parameter=limit>  -> [27, 15704, 28, 9226, 29]
        <parameter=offset> -> [27, 15704, 28, 3075, 29]
        </parameter>       -> [510, 15704, 29]
    (generic <parameter= is [27,15704,28]; the name token 9226=limit / 3075=offset is what
    distinguishes a read-window arg from any other parameter.) Only SUPERVISED positions
    (labels != -100) are tagged — matching where the denoise-stream CE actually applies.

The value region is [marker_end, next </parameter>). We tag the value tokens (the numeric
limit/offset the twin under-grounds), NOT the marker tokens.

CPU smoke: `python scripts/x1_read_window_arg_loss.py --smoke` builds a synthetic window with
one read_file(limit=..) call, one read_file(offset=..) call, and one file_path=.. call, and
asserts (1) ONLY the limit+offset value tokens are tagged, (2) the file_path value is NOT
tagged, (3) unsupervised (-100) copies of the same markers are NOT tagged, (4) a weight
tensor built from the mask raises exactly those positions to the read-window weight. No GPU.
"""
import os
import sys
import json
import argparse
from typing import List, Tuple


# default marker subsequences (Qwen3.5 native <parameter=NAME>value</parameter> format)
_DEFAULT_START_SEQS = "27,15704,28,9226,29;27,15704,28,3075,29"   # <parameter=limit> ; <parameter=offset>
_DEFAULT_END_SEQ = "510,15704,29"                                  # </parameter>


def _parse_seq(raw: str) -> Tuple[int, ...]:
    out = []
    for item in (raw or "").replace(" ", "").split(","):
        if item == "":
            continue
        out.append(int(item))
    return tuple(out)


def _parse_seq_list(raw: str) -> List[Tuple[int, ...]]:
    seqs = []
    for chunk in (raw or "").split(";"):
        s = _parse_seq(chunk)
        if s:
            seqs.append(s)
    return seqs


def read_window_markers():
    """(start_seqs, end_seq) from env (with defaults). start_seqs is a list of subsequences
    (one per read-window schema key), end_seq is the </parameter> subsequence."""
    start = _parse_seq_list(os.environ.get("FASTDLLM_READ_WINDOW_START_TOKEN_IDS", _DEFAULT_START_SEQS))
    end = _parse_seq(os.environ.get("FASTDLLM_READ_WINDOW_END_TOKEN_IDS", _DEFAULT_END_SEQ))
    return start, end


def _match_at(seq: List[int], sub: Tuple[int, ...], i: int) -> bool:
    m = len(sub)
    if i + m > len(seq):
        return False
    return tuple(seq[i:i + m]) == sub


def read_window_value_mask_row(ids: List[int], labels: List[int],
                               start_seqs: List[Tuple[int, ...]], end_seq: Tuple[int, ...]) -> List[bool]:
    """Per-token bool: True iff the token is a SUPERVISED value token inside a
    <parameter=limit>..</parameter> or <parameter=offset>..</parameter> region.

    A start-marker subsequence opens a value region at its end; the region runs until (and
    excluding) the next </parameter> subsequence (or EOS). Only positions with labels != -100
    are tagged (the CE only applies there; the denoise-stream deficit lives on emitted values).
    """
    n = len(ids)
    mask = [False] * n
    if not start_seqs or not end_seq:
        return mask
    i = 0
    while i < n:
        opened = None
        for s in start_seqs:
            if _match_at(ids, s, i):
                opened = len(s)
                break
        if opened is None:
            i += 1
            continue
        val_start = i + opened
        # find next end marker at or after val_start
        j = val_start
        val_end = n
        while j < n:
            if _match_at(ids, end_seq, j):
                val_end = j
                break
            j += 1
        for k in range(val_start, val_end):
            if labels is None or labels[k] != -100:
                mask[k] = True
        i = val_end + (len(end_seq) if val_end < n else 0)
    return mask


def read_window_active_mask(labels_tensor, ids_tensor=None):
    """Torch wrapper. labels_tensor / ids_tensor are [B, S] long tensors. Returns a bool
    tensor [B, S] of read-window supervised value positions. When ids_tensor is None we detect
    markers on `labels_tensor` (works when read-window value positions are supervised, i.e.
    labels==ids on the assistant emission — which is exactly the training regime here)."""
    import torch
    start_seqs, end_seq = read_window_markers()
    labs = labels_tensor.tolist()
    ids = ids_tensor.tolist() if ids_tensor is not None else labs
    B = len(labs)
    out = torch.zeros_like(labels_tensor, dtype=torch.bool)
    for b in range(B):
        # marker detection must run on the true token ids; -100 label positions cannot host a
        # marker subsequence (markers are structural tokens, always supervised on emission),
        # so detecting on ids (falling back to labels) is exact for the supervised regime.
        row_ids = ids[b]
        # replace any -100 in the id-view with a sentinel that never appears in a marker
        row_ids = [t if t >= 0 else -1 for t in row_ids]
        m = read_window_value_mask_row(row_ids, labs[b], start_seqs, end_seq)
        if any(m):
            out[b] = torch.tensor(m, dtype=torch.bool, device=labels_tensor.device)
    return out


# --------------------------------------------------------------------------------------------
# CPU unit smoke
# --------------------------------------------------------------------------------------------
def _smoke():
    import torch

    LIM = [27, 15704, 28, 9226, 29]    # <parameter=limit>
    OFF = [27, 15704, 28, 3075, 29]    # <parameter=offset>
    FP = [27, 15704, 28, 8100, 29]     # <parameter=file_path>  (8100 = arbitrary non-limit/offset name)
    END = [510, 15704, 29]             # </parameter>

    os.environ["FASTDLLM_READ_WINDOW_START_TOKEN_IDS"] = _DEFAULT_START_SEQS
    os.environ["FASTDLLM_READ_WINDOW_END_TOKEN_IDS"] = _DEFAULT_END_SEQ

    lim_val = [321, 322]        # limit value tokens (supervised)
    off_val = [410]             # offset value tokens (supervised)
    fp_val = [900, 901, 902]    # file_path value tokens (must NOT be tagged)

    # Realistic regime: a tool-RESULT / prompt context (unsupervised, -100) that ECHOES a
    # read_file(limit=..) call, followed by the assistant EMISSION (fully supervised: the
    # assistant generates the whole tool call, so labels == ids across markers AND values).
    ctx_echo = LIM + [777] + END          # an unsupervised read-window arg in context
    ctx = [50, 51, 52] + ctx_echo + [60]  # all -100
    emit = LIM + lim_val + END + OFF + off_val + END + FP + fp_val + END
    ids = ctx + emit
    n = len(ids)

    labels = [-100] * len(ctx) + list(emit)   # emission fully supervised (labels == ids)
    assert len(labels) == n, (len(labels), n)

    start_seqs, end_seq = read_window_markers()
    m = read_window_value_mask_row(ids, labels, start_seqs, end_seq)
    tagged = [i for i, b in enumerate(m) if b]

    lim_start = len(ctx) + len(LIM)
    off_start = lim_start + len(lim_val) + len(END) + len(OFF)
    fp_start = off_start + len(off_val) + len(END) + len(FP)
    expect = list(range(lim_start, lim_start + len(lim_val))) + list(range(off_start, off_start + len(off_val)))

    assert tagged == expect, f"ASSERT1 read-window value tags wrong: got {tagged}, want {expect}"
    fp_positions = set(range(fp_start, fp_start + len(fp_val)))
    assert not (set(tagged) & fp_positions), "ASSERT2 file_path value tagged (should be limit/offset only)"

    # ASSERT3: the UNSUPERVISED context echo (value token 777 at -100) is NOT tagged
    echo_val_pos = 3 + len(LIM)  # position of 777 inside ctx
    assert not m[echo_val_pos], "ASSERT3 unsupervised context-echo read-window value was tagged"

    # ASSERT4: torch weight tensor raises exactly those positions to the read-window weight
    W = 5.0
    labt = torch.tensor([labels], dtype=torch.long)
    rw = read_window_active_mask(labt)
    base = torch.ones_like(labt, dtype=torch.float32)
    weights = torch.where(rw, torch.maximum(base, torch.full_like(base, W)), base)
    raised = (weights[0] > 1.0).nonzero().flatten().tolist()
    assert raised == expect, f"ASSERT4 weight tensor raised {raised}, want {expect}"
    assert torch.allclose(weights[0][expect], torch.full((len(expect),), W)), "ASSERT4 weight value != W"

    print(f"[x1-smoke] PASS tagged read-window values={tagged} (limit+offset only)")
    print(f"[x1-smoke] PASS file_path value NOT tagged (positions {sorted(fp_positions)})")
    print(f"[x1-smoke] PASS unsupervised marker NOT tagged")
    print(f"[x1-smoke] PASS weight tensor raises exactly {raised} to {W}")
    print("[x1-smoke] ALL ASSERTIONS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the CPU unit smoke")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
