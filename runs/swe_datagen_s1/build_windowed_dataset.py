#!/usr/bin/env python3
"""build_windowed_dataset.py -- ITERATION-2 SHAPE-CORRECTED rebuild: EPISODE
WINDOWING (not front-truncation) over the frozen keepers pool.

WHY (the #126 / C46 trajectory-shape deficit):
  Iteration-1 (`build_swe_sft_lmflow_pretok.py`) LEFT-truncated each episode to the
  trainer block (12288), keeping ONLY the final edit-and-verify turns. Measured
  assistant-label retention = 69.88% (911,531 / 1,304,354 target tokens); the model
  NEVER trained on the dropped EARLY/MID context-management turns. The C46 paired
  read flagged trajectory shape as a candidate deficit. Iteration-2 replaces the
  single front-truncated window with a set of SERVE-EXACT SLIDING WINDOWS that TILE
  the whole episode, so every assistant turn is trained AND early/mid/late context
  management is covered.

SCHEME (see swe_tuning_campaign_design.md, STATUS 2026-07-13 amendment, for the
authoritative spec). Per episode, rendered ONCE with the 9B SERVING chat_template
(reusing build_swe_sft_dataset.keeper_to_instance / conv_for_template -> the exact
same serve-exact input_ids + assistant_spans as `train_swe_sft.tokenized.jsonl`):

  * TURN BOUNDARIES = every `<|im_start|>` token position (+ len as final sentinel).
    Consecutive tool messages share ONE `<|im_start|>user` wrapper (template
    grouping) -> that whole run is ONE turn-block. A window is ALWAYS a contiguous
    slice `full_ids[w_start:w_end]` with w_start,w_end on turn boundaries, so NO
    message / tool-call / <tool_response> envelope is ever split, and the slice is
    byte-identical to the corresponding region of the full serve render (serve-exact;
    it is exactly what the served model sees at that read position under the
    campaign's `truncation_side=left`).

  * PREFIX-ANCHORED FORWARD TILING + bounded read-back. Window 1 = the PREFIX
    [0:<=BLOCK] (system + task + early turns -> the model trains on the fresh task ->
    first actions, mirroring serve where early turns see the full prefix). Each later
    window starts `--ctx-overlap` tokens before the previous window's end (snapped to a
    turn boundary): a bounded slab of loss-masked read-back context, then it owns the
    next run of assistant turns that fit its <=BLOCK budget. Window RIGHT edge = the
    last owned assistant `<|im_end|>` (windows end at a real generation boundary). Every
    window is <= BLOCK BY CONSTRUCTION (right edge = largest turn boundary <= w_start+
    BLOCK; block-fit guaranteed, not checked after the fact). Stride ~= (BLOCK -
    ctx_overlap) target tokens => #windows is BOUNDED (~ceil(episode/stride)), not the
    O(#turns) blow-up of maximal read-back. Each assistant turn is a TARGET in EXACTLY
    ONE window => union of targets = all assistant turns => ~100% retention (pre-cap).

  * LABEL POLICY (same as iteration-1): assistant spans only. A window's
    `assistant_spans` = ONLY the target turns it owns (remapped by -w_start). Turns
    that appear as read-back context (incl. assistant turns owned by an earlier
    window) are loss-masked (NOT in assistant_spans).

  * DEDUP / anti-domination. A long episode yields more windows (more rows) than a
    short one. Primary rule (works with the UNCHANGED trainer's uniform row sampler):
    CAP windows/episode at --max-windows; if an episode needs more, keep the FIRST
    (early) + LAST (late) window and evenly-spaced interior windows (stratified, so
    early/mid/late coverage survives the cap). Secondary/optional: every row carries
    `sample_weight = 1/(windows emitted for its episode)` so a future weighted trainer
    can exactly equalize per-episode without a rebuild (NOT required for this rebuild).

OUTPUT: `train_swe_sft_windowed.tokenized.jsonl` -- the SAME tokenized schema as
`train_swe_sft.tokenized.jsonl` (conversation_id / input_ids / assistant_spans /
n_tokens / n_label_tokens) plus window-metadata fields (episode_id, window_index,
n_windows_episode, window_position, target_asst_ordinals, includes_system,
sample_weight). Downstream `build_swe_sft_lmflow_pretok.py` reads only input_ids /
assistant_spans / conversation_id, so it consumes this file UNCHANGED (the
left-truncation there becomes a no-op since every window already <= BLOCK).

PRE-REGISTERED: the FINAL post-promotion rebuild is a MECHANICAL re-run of THIS
script (same seed, same BLOCK, same cap) on the updated keepers.jsonl -- no design
changes. CPU-only; deterministic; nothing external is read (windows only from keeper
episodes -> leakage no-op).
"""
import argparse
import bisect
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/mark/qwen_diffusion")
HERE = REPO / "runs/swe_datagen_s1"
sys.path.insert(0, str(HERE))
import build_swe_sft_dataset as B  # noqa: E402  reuse serve-exact render + firewall

from transformers import AutoTokenizer  # noqa: E402

OUT_DIR = REPO / "data/swe_sft_pool"
OUT_JSONL = OUT_DIR / "train_swe_sft_windowed.tokenized.jsonl"
AUDIT_JSON = OUT_DIR / "windowed_dataset_audit.json"
REPORT_MD = OUT_DIR / "windowed_dataset_report.md"

ITER1_RETENTION = 0.6988  # front-truncation @12288 (design STATUS 2026-07-09 later)
ITER1_KEPT = 911531
ITER1_TOTAL = 1304354


def third(frac):
    return "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")


def window_episode(ids, spans, block, im_start, ctx_overlap):
    """Tile one episode into serve-exact <=block windows via PREFIX-ANCHORED forward
    tiling with BOUNDED read-back. Window 1 = the PREFIX [0:<=block] (carries
    system+task+early turns, so the model trains on the fresh task -> first actions,
    mirroring serve where early turns always see the full prefix). Each subsequent
    window starts `ctx_overlap` tokens before the previous window's end (snapped to a
    turn boundary) -> a bounded slab of loss-masked read-back context, then owns the
    NEXT run of assistant turns that fit within its <=block budget. STRIDE per window
    ~= (block - ctx_overlap) target tokens, so #windows ~= ceil(episode_tokens / stride)
    -- bounded, not the O(#turns) blow-up of maximal read-back. Targets are DISJOINT
    across windows (each assistant turn owned once) -> ~100% retention. Every window is
    a contiguous boundary-aligned slice <=block (serve-exact, whole turns; no split
    envelopes). Returns (list-of-window-dicts, n_assistant_turns)."""
    n = len(ids)
    boundaries = [i for i, t in enumerate(ids) if t == im_start]
    boundaries.append(n)  # final sentinel
    at = []  # (enclosing_boundary_index, span_start, span_end) per assistant turn, in order
    for (s, e) in spans:
        b = bisect.bisect_right(boundaries, s) - 1
        at.append((b, int(s), int(e)))
    at.sort()
    M = len(at)
    windows = []
    owned = 0          # next unowned assistant-turn ordinal
    w_start = 0        # window-1 is the prefix
    while owned < M:
        # right edge: largest turn boundary with [w_start:w_end] <= block
        be = bisect.bisect_right(boundaries, w_start + block) - 1  # boundaries[be] <= w_start+block
        w_end = boundaries[be]
        # which unowned targets fall fully inside [w_start:w_end)?
        j = owned
        while j < M and at[j][2] <= w_end and at[j][1] >= w_start:
            j += 1
        if j == owned:
            # next target does not fit from this w_start -> jump the anchor to it
            # (serve-time left-truncation: read window = last <=block ending at its end).
            nb_end = boundaries[at[owned][0] + 1]
            w_start = boundaries[bisect.bisect_left(boundaries, nb_end - block)]
            continue
        w_end = boundaries[at[j - 1][0] + 1]  # snap right edge to last owned target's end
        assert w_end - w_start <= block, (w_start, w_end, block)
        win_ids = ids[w_start:w_end]
        win_spans = [[s - w_start, e - w_start] for (_, s, e) in at[owned:j]]
        tcenter = (at[owned][1] + w_end) / 2.0
        windows.append({
            "input_ids": win_ids,
            "assistant_spans": win_spans,
            "w_start": w_start,
            "w_end": w_end,
            "target_ordinals": [owned, j - 1],
            "target_tok_span": [at[owned][1], w_end],
            "pos_frac": tcenter / n if n else 0.0,
            "includes_system": w_start == 0,
        })
        owned = j
        if owned < M:
            # next window: BOUNDED read-back = ctx_overlap tokens before this w_end,
            # snapped to a turn boundary (<= the next unowned target's start).
            nb_start = boundaries[at[owned][0]]           # next target's turn start
            anchor = min(w_end - ctx_overlap, nb_start)   # never skip past the next target
            w_start = boundaries[bisect.bisect_left(boundaries, max(0, anchor))]
            if w_start > nb_start:                        # snap-up overshot the target: clamp
                w_start = nb_start
    return windows, M


def stratified_cap(windows, cap):
    """Keep <=cap windows, always retaining the first + last, interior evenly spaced."""
    if len(windows) <= cap:
        return list(range(len(windows)))
    if cap == 1:
        return [len(windows) - 1]  # degenerate: keep the LATE window (iteration-1 behaviour)
    idx = [0]
    interior = cap - 2
    if interior > 0:
        span = len(windows) - 1
        for k in range(1, interior + 1):
            idx.append(round(k * span / (interior + 1)))
    idx.append(len(windows) - 1)
    return sorted(set(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=12288)
    ap.add_argument("--max-windows", type=int, default=6,
                    help="cap windows/episode (stratified subsample beyond it).")
    ap.add_argument("--ctx-overlap", type=int, default=3072,
                    help="read-back context tokens carried from the previous window "
                         "(loss-masked); stride ~= block - ctx_overlap.")
    ap.add_argument("--seed", type=int, default=71101)
    ap.add_argument("--audit-samples", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_JSONL))
    args = ap.parse_args()
    assert args.block % 32 == 0, "block must be divisible by bd_size 32 (pretok pad path)"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[build_windowed] {ts} block={args.block} max_windows={args.max_windows} seed={args.seed}")

    # ---- leakage firewall (KILL-D1), reused verbatim from iteration-1 builder ----
    holdout, sha = B.reconstruct_holdout()
    quarantined = set()
    if B.QUARANTINED.exists():
        for line in B.QUARANTINED.read_text().splitlines():
            if line.strip():
                quarantined.add(json.loads(line)["instance_id"])
    keepers = [json.loads(l) for l in B.KEEPERS.read_text().splitlines() if l.strip()]
    kept_ids = [k["instance_id"] for k in keepers]
    assert len(kept_ids) == len(set(kept_ids)), "duplicate keeper ids"
    leak = sorted(set(kept_ids) & holdout)
    if leak:
        raise SystemExit(f"KILL-D1 LEAK: {len(leak)} keeper ids in holdout: {leak[:10]}")
    q_in = sorted(set(kept_ids) & quarantined)
    if q_in:
        raise SystemExit(f"KILL: quarantined ids in keepers: {q_in}")
    print(f"[gate] holdout {len(holdout)} sha=={sha[:12]}.. ; keeper∩holdout=0 ; quarantine-in-train=0 ; keepers={len(keepers)}")

    tok = AutoTokenizer.from_pretrained(str(B.STUDENT_MODEL), trust_remote_code=True)
    serve = B.SERVE_TEMPLATE_FILE.read_text()
    IM_START = tok.convert_tokens_to_ids("<|im_start|>")
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    ASST = tok.encode("assistant\n", add_special_tokens=False)

    def scan_spans(ids):
        spans = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == IM_START and ids[i + 1:i + 1 + len(ASST)] == ASST:
                j = i + 1 + len(ASST)
                k = j
                while k < n and ids[k] != IM_END:
                    k += 1
                end = min(k, n - 1)
                spans.append((j, end + 1))
                i = end + 1
            else:
                i += 1
        return spans

    # ---- render every episode ONCE (serve-exact, untruncated) + window it ----
    out_rows = []
    ep_records = []          # per-episode audit
    total_labels = 0         # denominator: full-episode assistant target tokens
    emitted_labels = 0       # labels actually emitted across windows (post-cap)
    label_pos_iter2 = Counter()   # emitted label tokens by episode-position third
    label_pos_full = Counter()    # ALL label tokens by episode-position third (denominator)
    label_pos_iter1 = Counter()   # iteration-1 (last-BLOCK front-trunc) retained labels by third
    win_pos_hist = Counter()      # window count by position
    n_windows_hist = Counter()    # episodes by (pre-cap) window count
    capped_eps = 0
    labels_dropped_by_cap = 0
    eps_first_window_has_system = 0
    max_win_len = 0

    for idx, kp in enumerate(keepers):
        inst = B.keeper_to_instance(kp)
        conv = B.conv_for_template(inst)
        ids = tok.apply_chat_template(conversation=conv, tools=inst.get("tools"),
                                      chat_template=serve, add_generation_prompt=False,
                                      return_dict=True)["input_ids"]
        spans = scan_spans(ids)
        n = len(ids)
        ep_labels = sum(e - s for s, e in spans)
        total_labels += ep_labels
        # position thirds of ALL label tokens (denominator distribution)
        for s, e in spans:
            label_pos_full[third(((s + e) / 2.0) / n)] += (e - s)
        # iteration-1 replica: keep last BLOCK tokens, remap spans (front-truncation)
        cut = max(0, n - args.block)
        for s, e in spans:
            ns = max(cut, s)
            if e > ns:
                label_pos_iter1[third(((s + e) / 2.0) / n)] += (e - ns)

        windows, M = window_episode(ids, spans, args.block, IM_START, args.ctx_overlap)
        n_windows_hist[len(windows)] += 1
        keep_idx = stratified_cap(windows, args.max_windows)
        if len(keep_idx) < len(windows):
            capped_eps += 1
            dropped = set(range(len(windows))) - set(keep_idx)
            for di in dropped:
                labels_dropped_by_cap += sum(e - s for s, e in windows[di]["assistant_spans"])
        n_emit = len(keep_idx)
        teacher = B.TEACHER.get(kp["provenance"]["generator"], kp["provenance"]["generator"])
        first_emit_has_sys = False
        for wi, w in enumerate(keep_idx):
            win = windows[w]
            wlab = sum(e - s for s, e in win["assistant_spans"])
            emitted_labels += wlab
            max_win_len = max(max_win_len, len(win["input_ids"]))
            pos = "full" if len(windows) == 1 else third(win["pos_frac"])
            win_pos_hist[pos] += 1
            label_pos_iter2[third(win["pos_frac"])] += wlab
            if win["includes_system"] and wi == 0:
                first_emit_has_sys = True
            out_rows.append({
                "conversation_id": f"{kp['instance_id']}#w{wi}",
                "episode_id": kp["instance_id"],
                "input_ids": win["input_ids"],
                "assistant_spans": win["assistant_spans"],
                "n_tokens": len(win["input_ids"]),
                "n_label_tokens": wlab,
                "window_index": wi,
                "n_windows_episode": n_emit,
                "window_position": pos,
                "target_asst_ordinals": win["target_ordinals"],
                "includes_system": win["includes_system"],
                "sample_weight": round(1.0 / n_emit, 6),
                "teacher": teacher,
                "source": kp["source"],
                "repo": kp["repo"],
            })
        if first_emit_has_sys:
            eps_first_window_has_system += 1
        ep_records.append({
            "episode_id": kp["instance_id"], "n_tokens": n, "ep_labels": ep_labels,
            "n_assistant_turns": M, "n_windows_pre_cap": len(windows), "n_windows_emitted": n_emit,
        })
        if (idx + 1) % 100 == 0:
            print(f"  .. {idx+1}/{len(keepers)}  rows={len(out_rows)}")

    # ---- write dataset ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    ds_sha = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    try:
        out_rel = str(Path(args.out).resolve().relative_to(REPO))
    except ValueError:
        out_rel = str(args.out)

    # ---- serve-exact spot-audit: re-render N random episodes, prove windows are
    #      exact boundary-aligned slices of the serve render + spans wrap assistants ----
    rng = random.Random(args.seed)
    audit_eps = rng.sample(range(len(keepers)), min(args.audit_samples, len(keepers)))
    audit_details = []
    audit_pass = True
    for ei in audit_eps:
        kp = keepers[ei]
        inst = B.keeper_to_instance(kp)
        conv = B.conv_for_template(inst)
        ids = tok.apply_chat_template(conversation=conv, tools=inst.get("tools"),
                                      chat_template=serve, add_generation_prompt=False,
                                      return_dict=True)["input_ids"]
        spans = scan_spans(ids)
        windows, M = window_episode(ids, spans, args.block, IM_START, args.ctx_overlap)
        keep_idx = stratified_cap(windows, args.max_windows)
        w = windows[rng.choice(keep_idx)]
        ws, we = w["w_start"], w["w_end"]
        slice_ok = (w["input_ids"] == ids[ws:we])
        boundary_ok = (ws == 0 or ids[ws] == IM_START) and (we == len(ids) or ids[we] == IM_START)
        # byte-identity of decoded window vs decoded source slice
        byte_ok = (tok.decode(w["input_ids"]) == tok.decode(ids[ws:we]))
        # every assistant_span wraps a real assistant turn (header + closing im_end)
        span_ok = True
        for (s, e) in w["assistant_spans"]:
            gs, ge = s + ws, e + ws
            if not (ids[gs - 1 - len(ASST)] == IM_START and ids[gs - len(ASST):gs] == ASST and ids[ge - 1] == IM_END):
                span_ok = False
        fit_ok = (we - ws) <= args.block
        ok = slice_ok and boundary_ok and byte_ok and span_ok and fit_ok
        audit_pass = audit_pass and ok
        audit_details.append({
            "episode_id": kp["instance_id"], "ep_tokens": len(ids), "n_windows": len(windows),
            "audited_window": {"w_start": ws, "w_end": we, "len": we - ws,
                               "n_spans": len(w["assistant_spans"]), "position": w.get("pos_frac")},
            "slice_is_exact_serve_slice": slice_ok, "boundaries_on_turn_edges": boundary_ok,
            "decoded_byte_identical": byte_ok, "spans_wrap_assistant_turns": span_ok,
            "window_within_block": fit_ok, "PASS": ok,
        })

    retention = emitted_labels / total_labels if total_labels else 0.0
    retention_precap = (emitted_labels + labels_dropped_by_cap) / total_labels if total_labels else 0.0
    over_block = sum(1 for r in out_rows if r["n_tokens"] > args.block)

    audit = {
        "artifact": "swe_sft_windowed_dataset",
        "built_at": ts,
        "built_by": "runs/swe_datagen_s1/build_windowed_dataset.py",
        "supersedes": "iteration-1 front-truncation (build_swe_sft_lmflow_pretok.py @ max-len 12288)",
        "config": {"block": args.block, "max_windows_per_episode": args.max_windows,
                   "seed": args.seed, "label_policy": "assistant spans only (== iteration-1)"},
        "source_pool": {"keepers_jsonl": str(B.KEEPERS.relative_to(REPO)), "keepers_count": len(keepers)},
        "leakage_firewall": {
            "eval_holdout_ids": len(holdout), "eval_holdout_sha256": sha,
            "sha_asserted_equal": True, "keeper_x_holdout_overlap": 0,
            "quarantined_present_in_train": 0,
            "windows_from_keeper_episodes_only": all(r["episode_id"] in set(kept_ids) for r in out_rows),
            "external_text": "NONE (windows are boundary-aligned slices of keeper renders)",
        },
        "outputs": {
            "windowed_tokenized_jsonl": out_rel,
            "sha256": ds_sha, "n_rows_windows": len(out_rows), "n_episodes": len(keepers),
        },
        "retention": {
            "iteration1_front_trunc_pct": round(100 * ITER1_RETENTION, 2),
            "iteration1_kept_over_total": [ITER1_KEPT, ITER1_TOTAL],
            "iteration2_total_episode_labels": total_labels,
            "iteration2_emitted_labels": emitted_labels,
            "iteration2_retention_pct": round(100 * retention, 3),
            "iteration2_retention_precap_pct": round(100 * retention_precap, 3),
            "beats_iteration1": retention > ITER1_RETENTION,
            "delta_pp_vs_iteration1": round(100 * (retention - ITER1_RETENTION), 2),
        },
        "window_counts": {
            "total_windows": len(out_rows),
            "windows_per_episode_precap_hist": dict(sorted(n_windows_hist.items())),
            "mean_windows_per_episode": round(len(out_rows) / len(keepers), 3),
            "max_windows_emitted": max(r["n_windows_episode"] for r in out_rows),
            "episodes_capped": capped_eps, "labels_dropped_by_cap": labels_dropped_by_cap,
        },
        "position_histogram": {
            "window_position_counts": dict(win_pos_hist),
            "emitted_label_tokens_by_third_iter2": dict(label_pos_iter2),
            "all_label_tokens_by_third_full_episode": dict(label_pos_full),
            "retained_label_tokens_by_third_iter1_fronttrunc": dict(label_pos_iter1),
            "note": "iter1 front-truncation concentrates retained labels in LATE; iter2 windows "
                    "recover EARLY+MID coverage (compare the two by-third distributions).",
        },
        "block_fit": {
            "block": args.block, "windows_over_block": over_block, "max_window_len": max_win_len,
            "assertion": "0 windows exceed block (guaranteed by construction)", "PASS": over_block == 0,
        },
        "serve_exact_spot_audit": {
            "n_audited": len(audit_details), "all_pass": audit_pass,
            "checks": ["slice_is_exact_serve_slice", "boundaries_on_turn_edges",
                       "decoded_byte_identical", "spans_wrap_assistant_turns", "window_within_block"],
            "details": audit_details,
        },
        "system_coverage": {
            "episodes_with_system_in_first_window": eps_first_window_has_system,
            "pct": round(100 * eps_first_window_has_system / len(keepers), 1),
            "note": "fraction of episodes whose EARLIEST emitted window still carries the "
                    "system+task prompt (loss-masked). iteration-1 kept only the LATE window "
                    "and dropped system+task on all 328 truncated episodes.",
        },
        "dedup_weighting": {
            "primary_rule": "CAP windows/episode (stratified subsample) -> bounds row-count "
                            "inflation under the trainer's uniform row sampler.",
            "secondary_rule": "sample_weight=1/n_windows per row for an optional weighted trainer "
                              "(NOT required; trainer consumes input_ids+assistant_spans unchanged).",
        },
        "downstream": {
            "consumer": "runs/swe_datagen_s1/build_swe_sft_lmflow_pretok.py --tokenized <this> "
                        "--max-len 12288 (left-trunc becomes a no-op since windows <= block).",
            "trainer": "scripts/swe_sft_arm1_qlora_train.py (unchanged).",
        },
        "final_rebuild_preregistration": "MECHANICAL re-run of this exact script (same seed/block/cap) "
                                         "on the post-promotion keepers.jsonl; no design changes.",
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2))

    # ---- short report ----
    r = audit["retention"]; wc = audit["window_counts"]; ph = audit["position_histogram"]
    md = []
    md.append("# SWE-SFT windowed dataset (iteration-2, shape-corrected) — report\n")
    md.append(f"Built {ts} by `runs/swe_datagen_s1/build_windowed_dataset.py` "
              f"(block {args.block}, cap {args.max_windows}/episode, seed {args.seed}) from "
              f"`{audit['source_pool']['keepers_jsonl']}` ({len(keepers)} keepers).\n")
    md.append("## Label retention (the headline vs iteration-1)\n")
    md.append(f"- iteration-1 front-truncation @12288: **{r['iteration1_front_trunc_pct']}%** "
              f"({ITER1_KEPT:,}/{ITER1_TOTAL:,})")
    md.append(f"- iteration-2 windowing: **{r['iteration2_retention_pct']}%** "
              f"({emitted_labels:,}/{total_labels:,}); pre-cap {r['iteration2_retention_precap_pct']}%")
    md.append(f"- **delta: +{r['delta_pp_vs_iteration1']} pp** (beats 69.9%: {r['beats_iteration1']})\n")
    md.append("## Window counts\n")
    md.append(f"- total windows: **{wc['total_windows']}** across {len(keepers)} episodes "
              f"(mean {wc['mean_windows_per_episode']}/episode, max {wc['max_windows_emitted']})")
    md.append(f"- windows/episode (pre-cap) histogram: {wc['windows_per_episode_precap_hist']}")
    md.append(f"- episodes hitting the cap: {wc['episodes_capped']} "
              f"(labels dropped by cap: {wc['labels_dropped_by_cap']})\n")
    md.append("## Window-position histogram (early/mid/late coverage)\n")
    md.append(f"- windows by position: {ph['window_position_counts']}")
    md.append(f"- **emitted** label tokens by episode-third (iter2): {ph['emitted_label_tokens_by_third_iter2']}")
    md.append(f"- retained label tokens by third — iter1 FRONT-TRUNC (late-skewed): "
              f"{ph['retained_label_tokens_by_third_iter1_fronttrunc']}")
    md.append(f"- all label tokens by third (full episode, denominator): "
              f"{ph['all_label_tokens_by_third_full_episode']}\n")
    md.append("## Serve-exact spot-audit\n")
    md.append(f"- audited {len(audit_details)} random windows; ALL-PASS = **{audit_pass}** "
              f"(exact-slice + turn-boundary + decoded-byte-identical + spans-wrap-assistant + block-fit)\n")
    md.append("## Block-fit\n")
    md.append(f"- windows over block {args.block}: **{over_block}** (max window len {max_win_len}) — "
              f"guaranteed <= block by construction\n")
    md.append("## Leakage (no-op check)\n")
    md.append(f"- holdout {len(holdout)} sha==pin; keeper∩holdout **0**; windows from keeper episodes "
              f"only (external text: NONE)\n")
    md.append("## System/task coverage recovered\n")
    md.append(f"- episodes whose earliest window still carries system+task (loss-masked): "
              f"**{eps_first_window_has_system}/{len(keepers)}** "
              f"({audit['system_coverage']['pct']}%) — iteration-1 dropped it on all 328 truncated episodes.\n")
    md.append("## Outputs\n")
    md.append(f"- dataset: `{audit['outputs']['windowed_tokenized_jsonl']}` (sha256 {ds_sha[:16]}..)")
    md.append(f"- audit: `data/swe_sft_pool/windowed_dataset_audit.json`")
    md.append(f"- report: `data/swe_sft_pool/windowed_dataset_report.md`\n")
    md.append("## Final rebuild (pre-registered)\n")
    md.append("- MECHANICAL re-run of this exact script (same seed/block/cap) on the post-promotion "
              "keepers.jsonl — no design changes.\n")
    REPORT_MD.write_text("\n".join(md))

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "keepers": len(keepers), "windows": len(out_rows),
        "retention_pct": r["iteration2_retention_pct"], "vs_iter1_pp": r["delta_pp_vs_iteration1"],
        "mean_windows": wc["mean_windows_per_episode"], "max_windows": wc["max_windows_emitted"],
        "windows_precap_hist": wc["windows_per_episode_precap_hist"],
        "window_pos": dict(win_pos_hist), "over_block": over_block, "max_win_len": max_win_len,
        "spot_audit_all_pass": audit_pass, "episodes_capped": capped_eps,
        "sys_in_first_window": eps_first_window_has_system,
    }, indent=2))
    print(f"\nwrote:\n  {args.out}\n  {AUDIT_JSON}\n  {REPORT_MD}")


if __name__ == "__main__":
    main()
