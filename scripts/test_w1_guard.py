#!/usr/bin/env python3
"""Unit tests for w1_guard — each gate + the W-0 deploy-class false-accept scenarios.

Run: .venv-fastdllm/bin/python scripts/test_w1_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_suffix_miner import SuffixCopyMiner, DraftResult
from w1_guard import decide_commit, GuardConfig, GuardDecision

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


def accept_all(c):
    return True, [1] * len(c)


def reject_all(c):
    return False, [0] * len(c)


def test_unambiguous_commits():
    print("[test] n_cand==1 unambiguous span commits the whole draft")
    ctx = [50, 1, 2, 3, 4, 5, 6, 70, 80, 90, 91, 92, 93, 94]   # "1 2 3" -> unique "4 5 6"
    m = SuffixCopyMiner(min_match=2, cand_cap=8); m.append_context(ctx)
    d = m.draft([99, 1, 2, 3], 3)
    check(d.n_cand == 1, f"expected n_cand==1 got {d.n_cand} cands={d.candidates}")
    dec = decide_commit(d, ctx, accept_all, None, GuardConfig(min_match=2))
    check(dec.reason == "committed" and dec.committed == (4, 5, 6),
          f"unexpected {dec.reason} {dec.committed}")


def test_ambiguous_commits_only_common_prefix():
    print("[test] n_cand>=2 commits ONLY the maximal common prefix (near-dup tail never commits)")
    # two sources of "1 2 3": one -> (7,7,9,0), one -> (7,7,8,0). common prefix (7,7).
    ctx = [1, 2, 3, 7, 7, 9, 0, 55, 1, 2, 3, 7, 7, 8, 0, 66]
    m = SuffixCopyMiner(min_match=2, cand_cap=8); m.append_context(ctx)
    d = m.draft([42, 1, 2, 3], 4)
    check(d.n_cand == 2, f"expected 2 cands got {d.n_cand}: {d.candidates}")
    check(d.common_prefix == (7, 7), f"common_prefix {d.common_prefix}")
    dec = decide_commit(d, ctx, accept_all, None, GuardConfig(min_match=2))
    check(dec.reason == "committed" and dec.committed == (7, 7),
          f"should commit only common prefix, got {dec.reason} {dec.committed}")


def test_verify_reject_routes_k1():
    print("[test] verify reject => K=1 (G1)")
    ctx = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    m = SuffixCopyMiner(min_match=2, cand_cap=8); m.append_context(ctx)
    d = m.draft([9, 1, 2, 3], 3)
    dec = decide_commit(d, ctx, reject_all, None, GuardConfig(min_match=2))
    check(dec.commit_len == 0 and dec.reason == "k1_verify_reject", f"{dec.reason}")


def test_2nd_candidate_margin():
    print("[test] 2nd-candidate margin: runner-up also accepts a DIVERGENT full span => K=1")
    # Force n_cand==1 mining but simulate a divergent runner-up by hand-building a DraftResult where
    # the winner is full-length and a second candidate diverges yet 'accepts' under the mock verify.
    d = DraftResult(match_len=4, seed=(1, 2, 3, 4),
                    candidates=[(5, 6, 7), (5, 9, 9)],   # diverge at pos1
                    src_positions=[100, 20], src_dist=[10, 90],
                    common_prefix=(5,), n_cand=2)
    # With n_cand>=2 the guard commits only the common prefix (5,). Verify accepts everything.
    ctx_a = [0] * 100 + [5, 6, 7] + [0] * 97      # context[100]==5 (winner's source)
    dec = decide_commit(d, context=ctx_a, verify_span=accept_all, grammar_clip=None,
                        cfg=GuardConfig(min_match=2))
    check(dec.committed == (5,), f"ambiguous => commit common prefix only, got {dec.committed}")
    # Now n_cand==1 but a genuine runner-up (passed via candidates) that diverges & accepts.
    context = [0] * 100 + [5, 6, 7] + [0] * 97      # context[100:103] == winner (5,6,7)
    d1 = DraftResult(match_len=4, seed=(1, 2, 3, 4), candidates=[(5, 6, 7), (8, 9, 9)],
                     src_positions=[100, 20], src_dist=[10, 90], common_prefix=(5, 6, 7), n_cand=1)
    # n_cand==1 path commits candidates[0] fully; the divergent runner-up accepting trips the margin.
    def verify_margin(c):
        return True, [1] * len(c)      # BOTH accept => runner_up_accepted => K=1
    dec1 = decide_commit(d1, context=context, verify_span=verify_margin, grammar_clip=None,
                         cfg=GuardConfig(min_match=2))
    check(dec1.commit_len == 0 and dec1.reason == "k1_margin",
          f"runner-up accept should trip margin, got {dec1.reason} {dec1.commit_len}")


def test_source_distance_gate():
    print("[test] source-distance gate: ambiguous AND all sources far => K=1 (G3)")
    d = DraftResult(match_len=5, seed=(1, 2, 3, 4, 5), candidates=[(9, 9), (8, 8)],
                    src_positions=[500, 100], src_dist=[2000, 3000],   # both beyond d_safe
                    common_prefix=(), n_cand=2)
    dec = decide_commit(d, context=[0] * 4000, verify_span=accept_all, grammar_clip=None,
                        cfg=GuardConfig(min_match=2, d_safe=1024))
    check(dec.commit_len == 0 and dec.reason == "k1_ambig_far", f"{dec.reason}")
    # a near ambiguous span with an empty common prefix falls through to no-commit as well
    d2 = DraftResult(match_len=5, seed=(1, 2, 3, 4, 5), candidates=[(9, 9), (8, 8)],
                     src_positions=[3900, 3800], src_dist=[100, 200], common_prefix=(), n_cand=2)
    dec2 = decide_commit(d2, context=[0] * 4000, verify_span=accept_all, grammar_clip=None,
                         cfg=GuardConfig(min_match=2, d_safe=1024))
    check(dec2.commit_len == 0, f"empty common prefix near => no commit, got {dec2.commit_len}")


def test_truncated_routes_k1():
    print("[test] occurrence-scan truncation => K=1 (ambiguity unknown)")
    d = DraftResult(match_len=6, seed=(1,) * 6, candidates=[(2, 3)], src_positions=[10],
                    src_dist=[5], common_prefix=(2, 3), n_cand=1, truncated=True)
    dec = decide_commit(d, context=[0] * 100, verify_span=accept_all, grammar_clip=None,
                        cfg=GuardConfig(min_match=2))
    check(dec.commit_len == 0 and dec.reason == "k1_truncated", f"{dec.reason}")


def test_grammar_clip():
    print("[test] FSM clip shortens/empties the committed prefix (G5)")
    ctx = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]
    m = SuffixCopyMiner(min_match=2, cand_cap=8); m.append_context(ctx)
    d = m.draft([9, 1, 2, 3], 4)
    # grammar allows only the first 2 tokens (e.g. </parameter> boundary)
    dec = decide_commit(d, ctx, accept_all, grammar_clip=lambda c: list(c[:2]),
                        cfg=GuardConfig(min_match=2))
    check(dec.commit_len == 2, f"grammar clip to 2, got {dec.commit_len}")
    dec2 = decide_commit(d, ctx, accept_all, grammar_clip=lambda c: [], cfg=GuardConfig(min_match=2))
    check(dec2.commit_len == 0 and dec2.reason == "k1_grammar_empty", f"{dec2.reason}")


def test_copy_assert():
    print("[test] byte-equal copy-assert raises on corruption (G4)")
    d = DraftResult(match_len=4, seed=(1, 2, 3, 4), candidates=[(5, 6, 7)], src_positions=[0],
                    src_dist=[10], common_prefix=(5, 6, 7), n_cand=1)
    raised = False
    try:
        decide_commit(d, context=[9, 9, 9, 9], verify_span=accept_all, grammar_clip=None,
                      cfg=GuardConfig(min_match=2))   # context[0:3]=[9,9,9] != committed [5,6,7]
    except AssertionError:
        raised = True
    check(raised, "copy-assert should raise on context mismatch")


def test_no_draft_floor():
    print("[test] no match => K=1 floor, never lengthens decode")
    ctx = [1, 2, 3, 4, 5]
    m = SuffixCopyMiner(min_match=4, cand_cap=8); m.append_context(ctx)
    d = m.draft([90, 91, 92], 3)      # no 4-gram match
    dec = decide_commit(d, ctx, accept_all, None, GuardConfig(min_match=4))
    check(dec.commit_len == 0 and dec.reason == "k1_no_draft", f"{dec.reason}")


if __name__ == "__main__":
    test_unambiguous_commits()
    test_ambiguous_commits_only_common_prefix()
    test_verify_reject_routes_k1()
    test_2nd_candidate_margin()
    test_source_distance_gate()
    test_truncated_routes_k1()
    test_grammar_clip()
    test_copy_assert()
    test_no_draft_floor()
    print(f"\n=== w1_guard: {PASS} checks passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
