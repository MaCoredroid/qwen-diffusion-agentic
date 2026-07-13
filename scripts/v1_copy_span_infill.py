#!/usr/bin/env python
# coding=utf-8
"""V1 — copy-span joint-infill consistency (training, folded into the FLARE two-stream conversion).

SPEC: k_raise_campaign_design.md SECTION V.1 (commits da025e0 / 276f3bf, DIRECTIVE-3 piggyback).

This is the NEW training-config/loss file that carries the ENTIRE V1 objective. It is imported by an
ENV-GATED, strict-no-op hook in the shared finetuner (activated only when FASTDLLM_V1_COPY_SPAN=1), so
the plain conversion path (arm A) is byte-identical and never enters any V1 code. arm A does NOT set the
flag; this module is inert for it.

WHAT V1 PRESCRIBES (implemented here, exactly):
  * Extend L_diff with a whole-copy-span joint-infill target: for a keeper assistant turn, identify its
    arg-value COPY spans; mask ALL L positions of a sampled span SIMULTANEOUSLY in the denoise stream,
    keep the entire prior context (which by construction contains the verbatim source string) clean, and
    supervise all L masked positions jointly from a single forward (joint CE == the diff loss over those
    positions). This is realized here by INJECTING a per-window `flare_mask_indices` into the existing
    two-stream forward hook (models/.../modeling.py accepts `flare_mask_indices`); view0 masks exactly the
    union {random block noise} ∪ {whole copy spans} ∪ {derived-value forced}, view1 is the complement, and
    _compute_flare_losses already yields the joint CE over all masked positions. NO modeling.py edit.
  * Span identification reuses the census 4-gram copy predicate (runs/k_census: tok_ngrams n=4, is_copy =
    trailing 4-gram present in the earlier same-doc context set) and TIGHTENS it for precision: a training
    copy span must be a contiguous substring of a SINGLE earlier source location (longest verbatim match),
    not the union n-gram set. Reported as a loose->tight shrinkage line in the V1 manifest.
  * Span-length curriculum (mirrors O3): L in {2,3,4} for the first ~1/3 of the conversion, ceiling ->8,
    then ->16+ (we use 32) over the remainder. Cap masked span length by the ceiling (<=32) so no single
    microbatch is dominated by one canvas-scale run (this IS the min(L,32) weighting realized as a mask cap).
  * Loss composition: copy vs derived value tokens PARTITION (never both). Derived ARG_VALUE / FREETEXT stay
    on the plain path (forced-masked prob 1.0 + VALUE_SPAN_LOSS_WEIGHT). Copy spans get the whole-span joint
    masking under the curriculum.

LOGGED CONSERVATIVE CHOICES (design leaves these open; picked conservatively, per task directive):
  [C1] Copy-span loss WEIGHT: the spec says "put L_copy at the O1 joint-commit weight" (the default diff
       weight 1.0) while derived stays 2.0. We KEEP copy positions at the plain VALUE_SPAN_LOSS_WEIGHT=2.0
       (via the unchanged token-id loss-weight path) rather than down-weighting to 1.0. This is the
       conservative choice: it changes NOTHING on the derived path (zero regression risk to the certified
       exact_args / KILL-T1 guard), and keeps arm B a strict superset perturbation of the plain recipe
       (plain + guaranteed whole-copy-span joint masking + curriculum). Down-weighting copy to 1.0 is the
       more aggressive edit and is a clean follow-up if V1 shows lift.
  [C2] Curriculum step source: the collator increments an internal per-call counter; with GRAD_ACCUM=1 and
       batch==1 this equals the optimizer global_step. progress = counter / MAX_STEPS.
  [C3] Per-span masking: for a tight copy span of true length Ls and curriculum ceiling C, we mask a
       contiguous sub-window of length L = min(Ls, C) anchored at the span start (whole span if Ls<=C).
  [C4] The random-block component of flare_mask_indices is regenerated here (bd-block rates ~ U[min,max],
       adaptive low-rate on non-value blocks) so L_diff on non-copy positions is preserved when the model's
       own random sampler is bypassed by the provided mask. Distributional (not bit-identical) fidelity to
       the plain recipe is sufficient for this experimental arm.

CPU smoke: `python scripts/v1_copy_span_infill.py --smoke` builds one synthetic batch and asserts the span
mask selects WHOLE spans (no partial-span holes) and the joint loss is finite. No GPU, no model load.
"""
import os
import sys
import json
import argparse
from typing import List, Tuple, Optional

NGRAM = 4


# --------------------------------------------------------------------------------------------------
# env parsing (mirror models/.../modeling.py argument-span settings)
# --------------------------------------------------------------------------------------------------
def _parse_int_list(raw: str) -> Tuple[int, ...]:
    out = []
    for item in (raw or "").replace(";", ",").replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return tuple(out)


def arg_span_marker_ids():
    """(start_ids, end_ids) subsequences for <parameter=...> / </parameter>, as the recipe exports them."""
    start = _parse_int_list(os.environ.get("FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS", ""))
    end = _parse_int_list(os.environ.get("FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS", ""))
    return start, end


def _find_subseq(seq: List[int], sub: Tuple[int, ...], start: int) -> int:
    """Index of first occurrence of `sub` in seq at or after `start`, else -1."""
    if not sub:
        return -1
    n, m = len(seq), len(sub)
    for i in range(start, n - m + 1):
        if tuple(seq[i:i + m]) == sub:
            return i
    return -1


def argvalue_region_mask(ids: List[int], start_ids: Tuple[int, ...], end_ids: Tuple[int, ...]) -> List[bool]:
    """Per-token bool: True iff strictly inside a <parameter=...> ... </parameter> value region.

    Mirrors modeling._argument_span_active_mask semantics at token granularity: after a start-marker
    subsequence ends, tokens are ARG_VALUE until (and excluding) the next end-marker subsequence.
    """
    n = len(ids)
    inside = [False] * n
    if not start_ids or not end_ids:
        return inside
    i = 0
    while i < n:
        s = _find_subseq(ids, start_ids, i)
        if s < 0:
            break
        val_start = s + len(start_ids)
        e = _find_subseq(ids, end_ids, val_start)
        if e < 0:
            for k in range(val_start, n):
                inside[k] = True
            break
        for k in range(val_start, e):
            inside[k] = True
        i = e + len(end_ids)
    return inside


# --------------------------------------------------------------------------------------------------
# census 4-gram copy predicate (reused) + V1 tight precision fix
# --------------------------------------------------------------------------------------------------
def tok_ngrams(ids, n=NGRAM):
    s = set()
    for i in range(len(ids) - n + 1):
        s.add(tuple(ids[i:i + n]))
    return s


def _is_contiguous_substring(needle: List[int], haystack: List[int]) -> bool:
    m = len(needle)
    if m == 0:
        return True
    if m > len(haystack):
        return False
    t = tuple(needle)
    for i in range(len(haystack) - m + 1):
        if tuple(haystack[i:i + m]) == t:
            return True
    return False


def loose_copy_runs(ids: List[int], av_mask: List[bool], doc_start: int, pos_start: int, pos_end: int):
    """CENSUS LOOSE predicate: maximal contiguous ARG_VALUE runs where is_copy(j) = trailing 4-gram in the
    earlier same-doc context set. Returns list of (s, e) half-open index runs. pos_start..pos_end is the
    supervised (label!=-100) window in doc-local coords; doc_start is the doc's first index in `ids`.
    """
    ctx = ids[doc_start:pos_start]              # earlier same-doc context tokens
    ctx_set = tok_ngrams(ctx, NGRAM) if len(ctx) >= NGRAM else set()
    runs = []
    cur_s = None
    for j in range(pos_start, pos_end):
        four = tuple(ids[j - NGRAM + 1:j + 1]) if j - doc_start >= NGRAM - 1 else None
        is_copy = av_mask[j] and (four is not None) and (four in ctx_set or four in tok_ngrams(ids[doc_start:j + 1], NGRAM))
        # grow ctx_set incrementally so copies can point at earlier ARG_VALUE emissions too
        if is_copy:
            if cur_s is None:
                cur_s = j
        else:
            if cur_s is not None:
                runs.append((cur_s, j))
                cur_s = None
    if cur_s is not None:
        runs.append((cur_s, pos_end))
    return runs, ctx


def tight_copy_spans(ids: List[int], av_mask: List[bool], doc_start: int, pos_start: int, pos_end: int,
                     min_len: int = 2):
    """V1 TIGHT predicate: trim each loose run to the maximal contiguous SUB-run that is a verbatim
    contiguous substring of the earlier same-doc source (single source location). Returns (spans, stats).
    """
    loose, ctx = loose_copy_runs(ids, av_mask, doc_start, pos_start, pos_end)
    tight = []
    n_loose_tok = sum(e - s for s, e in loose)
    n_tight_tok = 0
    for (s, e) in loose:
        # source pool = all earlier same-doc tokens (context + already-emitted assistant tokens)
        # slide a maximal verbatim window: for each start a in [s,e), extend b while ids[a:b] is a
        # contiguous substring of ids[doc_start:a] (strictly-earlier source), keep the longest.
        a = s
        while a < e:
            source = ids[doc_start:a]
            b = a + 1
            best_b = a  # exclusive
            # need at least min_len and a real earlier source occurrence
            while b <= e:
                if _is_contiguous_substring(ids[a:b], source):
                    best_b = b
                    b += 1
                else:
                    break
            if best_b - a >= min_len:
                tight.append((a, best_b))
                n_tight_tok += best_b - a
                a = best_b
            else:
                a += 1
    return tight, {
        "loose_runs": len(loose),
        "tight_spans": len(tight),
        "loose_tokens": n_loose_tok,
        "tight_tokens": n_tight_tok,
    }


# --------------------------------------------------------------------------------------------------
# span-length curriculum
# --------------------------------------------------------------------------------------------------
def curriculum_ceiling(progress: float) -> int:
    """L in {2,3,4} for first ~1/3, ceiling ->8, then ->16+ (32). progress = step / max_steps in [0,1)."""
    if progress < 1.0 / 3.0:
        return 4
    if progress < 2.0 / 3.0:
        return 8
    return 32


def select_masked_span_positions(tight_spans: List[Tuple[int, int]], ceiling: int) -> List[int]:
    """[C3] For each tight copy span (s,e), mask a contiguous sub-window of length min(len, ceiling)
    anchored at s (whole span if it fits). Returns the flat list of masked positions."""
    positions = []
    for (s, e) in tight_spans:
        L = min(e - s, ceiling)
        positions.extend(range(s, s + L))
    return positions


# --------------------------------------------------------------------------------------------------
# per-window forced-mask construction (the object injected as flare_mask_indices)
# --------------------------------------------------------------------------------------------------
def _contiguous_docs(doc_ids_row: List[int]):
    """Yield (doc_start, doc_end) half-open runs of equal, valid (>=0) doc id."""
    n = len(doc_ids_row)
    i = 0
    while i < n:
        d = doc_ids_row[i]
        if d < 0:
            i += 1
            continue
        j = i
        while j < n and doc_ids_row[j] == d:
            j += 1
        yield (i, j)
        i = j


def build_forced_masks_for_row(ids: List[int], labels: List[int], doc_ids: List[int],
                               start_ids, end_ids, progress: float):
    """Return (copy_mask, derived_value_mask, stats) as per-position bool lists for one window row.

    copy_mask          : whole curriculum-sampled copy-span positions (joint-infill target).
    derived_value_mask : ARG_VALUE supervised positions that are NOT copy (plain derived-value forcing).
    """
    n = len(ids)
    av = argvalue_region_mask(ids, start_ids, end_ids)
    copy_mask = [False] * n
    derived_mask = [False] * n
    ceiling = curriculum_ceiling(progress)
    agg = {"loose_runs": 0, "tight_spans": 0, "loose_tokens": 0, "tight_tokens": 0, "masked_copy_tokens": 0}
    for (ds, de) in _contiguous_docs(doc_ids):
        # supervised sub-window inside this doc
        sup = [k for k in range(ds, de) if labels[k] != -100]
        if not sup:
            continue
        ps, pe = sup[0], sup[-1] + 1
        spans, st = tight_copy_spans(ids, av, ds, ps, pe)
        for kk in agg:
            if kk in st:
                agg[kk] += st[kk]
        copy_positions = set(select_masked_span_positions(spans, ceiling))
        for p in copy_positions:
            if labels[p] != -100:
                copy_mask[p] = True
        agg["masked_copy_tokens"] += sum(1 for p in copy_positions if labels[p] != -100)
        # derived value = ARG_VALUE supervised, not a copy-mask position
        for k in range(ps, pe):
            if av[k] and labels[k] != -100 and not copy_mask[k]:
                derived_mask[k] = True
    return copy_mask, derived_mask, agg


# --------------------------------------------------------------------------------------------------
# collator wrapper (env-gated entry point; imported by finetuner only when FASTDLLM_V1_COPY_SPAN=1)
# --------------------------------------------------------------------------------------------------
class CopySpanCollator:
    """Wrap a base data collator; add `flare_mask_indices` = random-block ∪ whole-copy-spans ∪ derived-value.

    Requires torch at call time (training env). Curriculum keyed on an internal call counter [C2].
    Writes a one-shot V1 manifest (loose->tight shrinkage, masked-copy token counts) to $V1_MANIFEST_PATH.
    """

    def __init__(self, base_collator, *, max_steps: int, bd_size: int,
                 mask_rate_min: float, mask_rate_max: float,
                 low_rate_min: float, low_rate_max: float,
                 manifest_path: Optional[str] = None, seed: int = 71101):
        self.base = base_collator
        self.max_steps = max(1, int(max_steps))
        self.bd_size = int(bd_size)
        self.mask_rate_min = float(mask_rate_min)
        self.mask_rate_max = float(mask_rate_max)
        self.low_rate_min = float(low_rate_min)
        self.low_rate_max = float(low_rate_max)
        self.manifest_path = manifest_path
        self.calls = 0
        self.seed = int(seed)
        self._start_ids, self._end_ids = arg_span_marker_ids()
        self._manifest_written = False
        self._agg = {"windows": 0, "loose_runs": 0, "tight_spans": 0,
                     "loose_tokens": 0, "tight_tokens": 0, "masked_copy_tokens": 0}

    def _random_block_mask(self, torch, label_valid, forced_value_any, device):
        """Per bd-block rate ~ U[min,max] (low-rate on blocks with no forced value), bernoulli, & valid."""
        B, S = label_valid.shape
        nblk = S // self.bd_size
        rates = torch.rand((B, nblk), device=device) * (self.mask_rate_max - self.mask_rate_min) + self.mask_rate_min
        low = torch.rand((B, nblk), device=device) * (self.low_rate_max - self.low_rate_min) + self.low_rate_min
        blk_val = forced_value_any.reshape(B, nblk, self.bd_size).any(dim=-1)
        rates = torch.where(blk_val, rates, low)
        p = rates.unsqueeze(-1).expand(B, nblk, self.bd_size).reshape(B, S)
        rnd = torch.rand((B, S), device=device) < p
        return rnd & label_valid

    def __call__(self, features):
        batch = self.base(features)
        try:
            import torch
        except Exception:
            return batch
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        doc_ids = batch.get("doc_ids")
        device = input_ids.device
        B, S = input_ids.shape
        if doc_ids is None:
            doc_ids = torch.zeros_like(input_ids)
        progress = min(self.calls / self.max_steps, 0.999)
        ii = input_ids.tolist()
        ll = labels.tolist()
        dd = doc_ids.tolist()
        copy = torch.zeros((B, S), dtype=torch.bool)
        derived = torch.zeros((B, S), dtype=torch.bool)
        for r in range(B):
            cm, dm, agg = build_forced_masks_for_row(ii[r], ll[r], dd[r], self._start_ids, self._end_ids, progress)
            copy[r] = torch.tensor(cm, dtype=torch.bool)
            derived[r] = torch.tensor(dm, dtype=torch.bool)
            self._agg["windows"] += 1
            for k in ("loose_runs", "tight_spans", "loose_tokens", "tight_tokens", "masked_copy_tokens"):
                self._agg[k] += agg.get(k, 0)
        copy = copy.to(device)
        derived = derived.to(device)
        label_valid = labels != -100
        forced_value_any = copy | derived
        rnd = self._random_block_mask(torch, label_valid, forced_value_any, device)
        flare = (rnd | copy | derived) & label_valid
        batch["flare_mask_indices"] = flare
        self.calls += 1
        self._maybe_write_manifest(progress)
        return batch

    def _maybe_write_manifest(self, progress):
        if self._manifest_written or not self.manifest_path or self.calls < 8:
            return
        a = self._agg
        loose_tok = max(a["loose_tokens"], 1)
        payload = {
            "spec": "k_raise_campaign_design.md SECTION V.1 (V1 copy-span joint-infill)",
            "detector": "census 4-gram (tok_ngrams n=4) + V1 tight single-source contiguous-substring fix",
            "windows_seen": a["windows"],
            "loose_runs": a["loose_runs"],
            "tight_spans": a["tight_spans"],
            "loose_tokens": a["loose_tokens"],
            "tight_tokens": a["tight_tokens"],
            "loose_to_tight_token_retention": round(a["tight_tokens"] / loose_tok, 4),
            "masked_copy_tokens": a["masked_copy_tokens"],
            "curriculum_progress_at_write": round(progress, 4),
            "conservative_choices": {
                "C1_copy_weight": "kept at plain VALUE_SPAN_LOSS_WEIGHT (2.0), not down-weighted to 1.0",
                "C2_curriculum_step": "collator call counter / MAX_STEPS",
                "C3_per_span_mask": "contiguous min(Ls,ceiling) window anchored at span start",
                "C4_random_block": "regenerated U[min,max] with adaptive low-rate on non-value blocks",
            },
        }
        try:
            os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
            with open(self.manifest_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            self._manifest_written = True
            print("[v1-copy-span] manifest written -> " + self.manifest_path, flush=True)
        except Exception as exc:
            print(f"[v1-copy-span] manifest write failed: {exc}", flush=True)


def wrap_collator_with_copy_span(base_collator):
    """Factory used by the env-gated finetuner hook. Reads config from env."""
    max_steps = int(os.environ.get("MAX_STEPS", "400") or 400)
    bd_size = int(os.environ.get("TRAIN_BD_SIZE", os.environ.get("FASTDLLM_TRAIN_BD_SIZE", "32")) or 32)
    mn = float(os.environ.get("FASTDLLM_FLARE_MASK_RATE_MIN", "0.3"))
    mx = float(os.environ.get("FASTDLLM_FLARE_MASK_RATE_MAX", "0.8"))
    lmn = float(os.environ.get("FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MIN", "0.02"))
    lmx = float(os.environ.get("FASTDLLM_FLARE_HIGH_ENTROPY_MASK_RATE_MAX", "0.12"))
    manifest = os.environ.get("V1_MANIFEST_PATH", "").strip() or None
    seed = int(os.environ.get("SEED", "71101") or 71101)
    print(f"[v1-copy-span] ACTIVE: max_steps={max_steps} bd_size={bd_size} "
          f"mask_rate=[{mn},{mx}] low_rate=[{lmn},{lmx}] manifest={manifest}", flush=True)
    return CopySpanCollator(base_collator, max_steps=max_steps, bd_size=bd_size,
                            mask_rate_min=mn, mask_rate_max=mx, low_rate_min=lmn, low_rate_max=lmx,
                            manifest_path=manifest, seed=seed)


# --------------------------------------------------------------------------------------------------
# CPU unit smoke
# --------------------------------------------------------------------------------------------------
def _smoke():
    import torch  # noqa: F401 -- needed for the finite-loss check

    # Synthetic token vocabulary. Build a window: [ctx: read_file result containing a code path] then an
    # assistant tool call re-emitting that path verbatim inside <parameter=file_path> ... </parameter>.
    START = (27, 15704, 28)   # <parameter=...>  (matches recipe start ids)
    END = (510, 15704, 29)    # </parameter>
    os.environ["FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS"] = "27,15704,28"
    os.environ["FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS"] = "510,15704,29"

    # a verbatim "path" copy span of 6 tokens that appears earlier in context
    copy_span = [900, 901, 902, 903, 904, 905]
    ctx = [50, 51, 52] + copy_span + [60, 61, 62, 63]            # context contains the source string
    # assistant emission: <parameter=> <copy_span> </parameter> + a DERIVED value (not in ctx)
    derived_val = [7001, 7002, 7003]
    emit = list(START) + copy_span + list(END) + list(START) + derived_val + list(END)
    ids = ctx + emit

    n = len(ids)
    labels = [-100] * len(ctx) + [-100] * len(START) + copy_span + [-100] * len(END) \
             + [-100] * len(START) + derived_val + [-100] * len(END)
    assert len(labels) == n, (len(labels), n)
    doc_ids = [0] * n

    start_ids, end_ids = arg_span_marker_ids()
    av = argvalue_region_mask(ids, start_ids, end_ids)
    # sanity: the two value regions are exactly copy_span and derived_val positions
    val_positions = [i for i, b in enumerate(av) if b]
    cs_start = len(ctx) + len(START)
    dv_start = cs_start + len(copy_span) + len(END) + len(START)
    expect_val = list(range(cs_start, cs_start + len(copy_span))) + list(range(dv_start, dv_start + len(derived_val)))
    assert val_positions == expect_val, f"arg-value region mismatch: {val_positions} != {expect_val}"

    # curriculum in the FINAL third so ceiling >= span length (whole span must be selectable)
    copy_mask, derived_mask, agg = build_forced_masks_for_row(ids, labels, doc_ids, start_ids, end_ids, progress=0.9)

    masked_copy_positions = [i for i, b in enumerate(copy_mask) if b]
    expect_copy = list(range(cs_start, cs_start + len(copy_span)))
    # ASSERT 1: the span mask selects the WHOLE copy span, contiguously, no holes, and nothing else.
    assert masked_copy_positions == expect_copy, \
        f"copy mask is not the whole span: got {masked_copy_positions}, want {expect_copy}"
    # ASSERT 1b: no partial-span hole — masked positions are a single contiguous run
    assert masked_copy_positions == list(range(masked_copy_positions[0], masked_copy_positions[-1] + 1)), \
        "copy mask has a hole (partial span)"
    # ASSERT 2: derived value tokens are forced-masked and DISJOINT from copy (partition)
    masked_derived = [i for i, b in enumerate(derived_mask) if b]
    assert masked_derived == list(range(dv_start, dv_start + len(derived_val))), \
        f"derived mask wrong: {masked_derived}"
    assert not (set(masked_copy_positions) & set(masked_derived)), "copy/derived partition violated"

    # ASSERT 3: joint CE over the masked span positions from a single (random) forward is FINITE.
    vocab = 8000
    torch.manual_seed(0)
    logits = torch.randn(1, n, vocab)
    tgt = torch.tensor([[labels[i] if copy_mask[i] else -100 for i in range(n)]])
    # joint cross-entropy L_copy = -sum log p(v_i | ctx, mask[1..L]) over the masked span
    import torch.nn.functional as F
    lc = F.cross_entropy(logits.view(-1, vocab), tgt.view(-1), ignore_index=-100)
    assert torch.isfinite(lc).item(), "L_copy is not finite"

    # curriculum sanity: earlier third caps span length at 4 (< 6) -> only a length-4 sub-window masked
    cm_early, _, _ = build_forced_masks_for_row(ids, labels, doc_ids, start_ids, end_ids, progress=0.1)
    early_positions = [i for i, b in enumerate(cm_early) if b]
    assert early_positions == list(range(cs_start, cs_start + 4)), \
        f"curriculum early-ceiling(4) failed: {early_positions}"

    print("[v1-smoke] PASS whole_copy_span=%s (contiguous, no holes)" % (masked_copy_positions,))
    print("[v1-smoke] PASS derived_partition=%s (disjoint from copy)" % (masked_derived,))
    print("[v1-smoke] PASS curriculum early ceiling=4 -> masked=%s ; late ceiling=32 -> whole span" % (early_positions,))
    print("[v1-smoke] PASS L_copy finite = %.4f" % float(lc))
    print("[v1-smoke] detector stats (late): %s" % json.dumps(agg))
    print("[v1-smoke] ALL ASSERTIONS PASSED")
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
