#!/usr/bin/env python3
"""RUNG W-1 commit guard (SECTION W + DIRECTIVE-5/6 + STATUS W-0 disposition).

The LOAD-BEARING correctness gate. W-0 measured that a prob floor CANNOT fix the near-duplicate
false-accept class (worst FAs at p=1.0) -> the guard must be STRUCTURAL. This module implements the
pre-registered commit rule and returns, per span, exactly how many drafted tokens are FA-safe to
commit (0 => K=1 fallback, the current-behavior floor).

PRE-REGISTERED COMMIT RULE (derived from the W-0 raw records; see w1_remine_cert.py for the proof):

  A drafted span exposes, from the suffix-automaton miner (w1_suffix_miner):
    - candidates: ALL distinct continuations of the longest anchored suffix match, recency-first
    - common_prefix: the maximal prefix shared by EVERY source (byte-identical regardless of which
      source is "right" -> FA-safe by construction)
    - n_cand: true ambiguity (over all sources); truncated: occurrence scan overflowed

  The maximal FA-safe committable length is the COMMON PREFIX:
    * n_cand == 1  -> common_prefix == the unique continuation; commit the whole drafted span.
    * n_cand >= 2  -> commit ONLY the common prefix; STOP at the ambiguity boundary. Extending into
      the recency-winner's divergent tail is NOT verify-separable: W-0 case k315t28 is a near-dup
      that is recent AND uniquely whole-span-accepts while the emitted gold rejects (gold_rec_rank=1)
      -> no verify/margin signal distinguishes it; only n_cand>=2 flags it. So the divergent tail is
      never committed on recency alone.

  On top of the common-prefix length, ALL of these gates must pass or the span routes to K=1:
    G1  batched verify: the committed prefix whole-span-accepts (argmax==draft at every position),
        via one forward over the top-m recency candidate canvases (cap 8, DIRECTIVE-5).
    G2  2nd-candidate margin: no OTHER candidate that DIFFERS on the committed positions also
        whole-span-accepts them. (Structurally satisfied by the common-prefix commit; computed and
        exposed as the margin metric, and it is the operative gate whenever n_cand==1 would extend.)
    G3  source-distance gate: if n_cand>=2 AND every candidate source sits beyond D_SAFE (the
        1025-3072 band where W-0 (f) measured FAs concentrate and recency-hit drops to 0.795),
        route to K=1 -- ambiguous + distant is the peak false-accept band.
    G4  byte-equal copy-assert: the committed tokens byte-equal context[src:src+len] (true by
        construction for a copy miner; a mismatch is a hard corruption error, never silently
        downgraded).
    G5  FSM-completability: the committed prefix is clipped to a grammar-legal, block-bounded span
        (<= remaining block, clipped at </parameter>) so the draft is completable under the FSM.

Fallback is ALWAYS K=1. The guard never *lengthens* what an unmatched decode would do; it only
converts K=1 into a longer committed copy when every gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

try:
    from w1_suffix_miner import DraftResult
except ImportError:                       # allow `python scripts/w1_guard.py`
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from w1_suffix_miner import DraftResult


# verify_span(cand_tokens) -> (whole_span_accept, per_position_argmax_bits)
VerifyFn = Callable[[Sequence[int]], "tuple[bool, list[int]]"]
# grammar_clip(committed_prefix_tokens) -> FSM-legal, block-bounded prefix (possibly shorter/empty)
GrammarClipFn = Callable[[Sequence[int]], "list[int]"]


@dataclass
class GuardConfig:
    min_match: int = 4
    cand_cap: int = 8                     # DIRECTIVE-5 batched-verify moving window
    d_safe: int = 1024                    # below the 1025-3072 FA-concentration band (W-0 (f))
    require_verify: bool = True           # G1 (engine sets True; pure-mining cert can disable)


@dataclass
class GuardDecision:
    commit_len: int                       # number of drafted tokens committed (0 => K=1 fallback)
    committed: tuple = ()                 # the committed token sequence
    reason: str = "k1"                    # why: committed | k1_no_draft | k1_truncated | k1_ambig_far
                                          #      | k1_grammar_empty | k1_verify_reject | k1_margin
    n_cand: int = 0
    common_prefix_len: int = 0
    winner_src_dist: Optional[int] = None
    runner_up_accepted: bool = False      # G2 telemetry
    verify_forwards: int = 0              # batched-verify cost (forwards spent), for the cost model
    metrics: dict = field(default_factory=dict)


def decide_commit(draft: DraftResult,
                  context: Sequence[int],
                  verify_span: Optional[VerifyFn],
                  grammar_clip: Optional[GrammarClipFn],
                  cfg: GuardConfig) -> GuardDecision:
    """Return the FA-safe commit decision for one drafted span. Pure; the engine supplies the
    verify_span (batched forward) and grammar_clip (FSM) callbacks."""
    # ---- pre-gates: no actionable / unsafe-to-reason draft => K=1 ----
    if draft.match_len < cfg.min_match or not draft.candidates:
        return GuardDecision(0, reason="k1_no_draft", n_cand=draft.n_cand)
    if draft.truncated:
        return GuardDecision(0, reason="k1_truncated", n_cand=draft.n_cand)

    # ---- maximal FA-safe committable length = common prefix (== winner iff n_cand==1) ----
    if draft.n_cand == 1:
        committable = list(draft.candidates[0])
    else:
        committable = list(draft.common_prefix)

    winner_src = draft.src_positions[0] if draft.src_positions else None
    winner_dist = draft.src_dist[0] if draft.src_dist else None

    # ---- G3 source-distance gate: ambiguous AND all sources distant => K=1 ----
    if draft.n_cand >= 2 and draft.src_dist and min(draft.src_dist) > cfg.d_safe:
        return GuardDecision(0, reason="k1_ambig_far", n_cand=draft.n_cand,
                             common_prefix_len=len(draft.common_prefix), winner_src_dist=winner_dist)

    if not committable:
        # ambiguous with immediate divergence (empty common prefix) => nothing FA-safe to commit
        return GuardDecision(0, reason="k1_ambig_diverge" if draft.n_cand >= 2 else "k1_no_draft",
                             n_cand=draft.n_cand, common_prefix_len=0, winner_src_dist=winner_dist)

    # ---- G5 FSM-completability: clip to grammar-legal, block-bounded prefix ----
    if grammar_clip is not None:
        committable = list(grammar_clip(committable))
        if not committable:
            return GuardDecision(0, reason="k1_grammar_empty", n_cand=draft.n_cand,
                                 common_prefix_len=len(draft.common_prefix), winner_src_dist=winner_dist)

    L = len(committable)

    # ---- G4 byte-equal copy-assert (hard invariant) ----
    if winner_src is not None:
        src_slice = list(context[winner_src:winner_src + L])
        if src_slice != committable:
            raise AssertionError(
                f"copy-assert violated: committed {committable} != context[{winner_src}:{winner_src+L}]={src_slice}")

    # ---- G1 batched verify + G2 2nd-candidate margin ----
    verify_forwards = 0
    runner_up_accepted = False
    if cfg.require_verify and verify_span is not None:
        # winner (=committed prefix) must whole-span-accept
        w_ok, w_bits = verify_span(committable)
        verify_forwards = 1
        if not w_ok:
            return GuardDecision(0, reason="k1_verify_reject", n_cand=draft.n_cand,
                                 common_prefix_len=len(draft.common_prefix),
                                 winner_src_dist=winner_dist, verify_forwards=verify_forwards)
        # 2nd-candidate margin: any OTHER candidate that DIFFERS on the committed positions and also
        # whole-span-accepts THOSE positions => ambiguity the commit cannot resolve => K=1.
        for c in draft.verify_candidates(cfg.cand_cap)[1:]:
            if tuple(c[:L]) == tuple(committable):
                continue                  # shares the committed prefix -> not a competitor here
            r_ok, _ = verify_span(list(c[:L]))
            verify_forwards += 1
            if r_ok:
                runner_up_accepted = True
                break
        if runner_up_accepted:
            return GuardDecision(0, reason="k1_margin", n_cand=draft.n_cand,
                                 common_prefix_len=len(draft.common_prefix),
                                 winner_src_dist=winner_dist, runner_up_accepted=True,
                                 verify_forwards=verify_forwards)

    return GuardDecision(
        commit_len=L, committed=tuple(committable), reason="committed",
        n_cand=draft.n_cand, common_prefix_len=len(draft.common_prefix),
        winner_src_dist=winner_dist, runner_up_accepted=runner_up_accepted,
        verify_forwards=verify_forwards,
        metrics={"match_len": draft.match_len, "winner_src": winner_src})


if __name__ == "__main__":
    from w1_suffix_miner import SuffixCopyMiner
    ctx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 40, 41, 42]     # "1 2 3" -> {4,5,6,7,...} and {40,41,42}
    m = SuffixCopyMiner(min_match=2, cand_cap=8)
    m.append_context(ctx)
    d = m.draft([99, 1, 2, 3], draft_len=3)
    print("draft: n_cand", d.n_cand, "cands", d.candidates, "common_prefix", d.common_prefix)
    dec = decide_commit(d, ctx, verify_span=lambda c: (True, [1] * len(c)),
                        grammar_clip=None, cfg=GuardConfig(min_match=2))
    print("decision:", dec.reason, "commit_len", dec.commit_len, "committed", dec.committed)
