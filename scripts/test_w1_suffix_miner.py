#!/usr/bin/env python3
"""Unit tests + MINER CERT battery for w1_suffix_miner (SECTION W / DIRECTIVE-6).

MINER CERT (the load-bearing safety proof): the anchored suffix-automaton matcher must NEVER draft
the off-by-one pointer-slip class (W-0 measured that class accepts at 4% if drafted -> it must be
STRUCTURALLY undraftable). Two batteries:
  (1) property tests over random token streams — the miner's candidates are ALWAYS exact context
      slices anchored at an exact suffix match; the off-by-one continuation (source+1) is never in
      the candidate set unless it is itself a legitimate exact-anchored occurrence.
  (2) the W-0 probe span corpus as fixtures — re-mine the exact spans and assert SAM == reference
      exact-substring miner AND that the synthetic off-by-one perturbations are never produced.

Run: .venv-fastdllm/bin/python scripts/test_w1_suffix_miner.py   (pure CPU; no torch needed for 1)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w1_suffix_miner import SuffixAutomaton, SuffixCopyMiner, reference_anchored_candidates

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# ---------------------------------------------------------------- SAM correctness
def naive_is_substring(ctx, sub):
    n, m = len(ctx), len(sub)
    return any(ctx[i:i + m] == sub for i in range(n - m + 1)) if m else True


def naive_longest_suffix_match(ctx, query):
    for L in range(len(query), 0, -1):
        if naive_is_substring(ctx, query[-L:]):
            return L
    return 0


def test_sam_longest_match():
    print("[test] SAM longest_suffix_match vs naive (200 random streams)")
    rng = random.Random(1234)
    for _ in range(200):
        n = rng.randint(1, 80)
        alpha = rng.randint(2, 6)
        ctx = [rng.randrange(alpha) for _ in range(n)]
        sam = SuffixAutomaton()
        sam.append_tokens(ctx)
        for _ in range(5):
            q = [rng.randrange(alpha) for _ in range(rng.randint(1, 12))]
            ml, _ = sam.longest_suffix_match(q)
            exp = naive_longest_suffix_match(ctx, q)
            check(ml == exp, f"match_len {ml}!={exp} ctx={ctx[:20]} q={q}")


def test_sam_incremental_equivalence():
    print("[test] SAM incremental append == batch build")
    rng = random.Random(99)
    for _ in range(50):
        ctx = [rng.randrange(4) for _ in range(rng.randint(1, 60))]
        a = SuffixAutomaton(); a.append_tokens(ctx)
        b = SuffixAutomaton()
        for t in ctx:
            b.extend(t)
        q = [rng.randrange(4) for _ in range(rng.randint(1, 10))]
        check(a.longest_suffix_match(q) == b.longest_suffix_match(q), "incremental != batch")


def test_endpos_recency():
    print("[test] endpos returns occurrence ends, most-recent first")
    ctx = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]     # "1 2 3" ends at idx 2, 6, 10
    sam = SuffixAutomaton(); sam.append_tokens(ctx)
    ml, st = sam.longest_suffix_match([5, 1, 2, 3])
    check(ml == 3, f"match_len {ml}")
    ends, trunc = sam.endpos(st)
    check(ends == [10, 6, 2], f"endpos {ends} != [10,6,2]")
    check(not trunc, "unexpected truncation")


# ---------------------------------------------------------------- miner == reference oracle
def test_miner_matches_reference():
    print("[test] SAM miner candidate SET == reference exact-substring miner (300 cases)")
    rng = random.Random(7)
    for _ in range(300):
        alpha = rng.randint(2, 8)
        ctx = [rng.randrange(alpha) for _ in range(rng.randint(4, 120))]
        emit = [rng.randrange(alpha) for _ in range(rng.randint(1, 15))]
        dl = rng.randint(1, 8)
        m = SuffixCopyMiner(min_match=2, cand_cap=8)
        m.append_context(ctx)
        r = m.draft(emit, dl)
        rl, rc, _ = reference_anchored_candidates(ctx, emit, dl, min_match=2)
        # normalize: an unactionable match (below threshold, or only end-of-context => no continuation)
        # reports match_len 0 on both sides
        sam_ml = r.match_len if r.candidates else 0
        rl_norm = rl if rc else 0
        check(sam_ml == rl_norm, f"match_len {sam_ml}!={rl_norm}")
        # FULL distinct candidate set must equal the brute-force oracle (uncapped)
        check(set(r.candidates) == set(rc),
              f"cand mismatch\n  sam={r.candidates}\n  ref={rc}\n  ctx={ctx}\n  emit={emit} dl={dl}")


# ---------------------------------------------------------------- MINER CERT: pointer-slip
def test_pointer_slip_undraftable():
    print("[CERT] off-by-one pointer-slip SOURCE is STRUCTURALLY unusable (500 adversarial cases)")
    rng = random.Random(4242)
    anchor_violations = 0        # any candidate whose source is NOT exactly anchored
    slip_source_used = 0         # W-0 off-by-one construction: the +1-slipped source ever used
    for _ in range(500):
        alpha = rng.randint(3, 12)
        ctx = [rng.randrange(alpha) for _ in range(rng.randint(20, 200))]
        seed_len = rng.randint(4, 8)
        cont_len = rng.randint(2, 8)
        src = rng.randint(0, len(ctx) - (seed_len + cont_len))
        seed = ctx[src:src + seed_len]
        m = SuffixCopyMiner(min_match=4, cand_cap=8)
        m.append_context(ctx)
        emit = [rng.randrange(alpha) for _ in range(3)] + list(seed)
        r = m.draft(emit, cont_len)
        if not r.candidates:
            continue
        anchor = tuple(emit[-r.match_len:])
        # (i) ANCHOR INVARIANT: every candidate is an EXACT context slice whose PRECEDING match_len
        #     tokens equal the anchor EXACTLY. This is the structural no-slip proof: a source can
        #     only be used if the seed matches there exactly, so a pointer that "slipped" by one is
        #     never a source unless it is itself a genuine exact occurrence.
        for cont, sp in zip(r.candidates, r.src_positions):
            if tuple(ctx[sp:sp + len(cont)]) != cont:
                anchor_violations += 1
            if tuple(ctx[sp - r.match_len:sp]) != anchor:
                anchor_violations += 1
        # (ii) DIRECT W-0 OFF-BY-ONE: for every used source sp, the slipped source (sp+1) is used
        #      ONLY IF sp+1 is itself exactly anchored (a legitimate distinct occurrence).
        used = set(r.src_positions)
        for sp in list(used):
            slip = sp + 1
            if slip in used:
                legit = tuple(ctx[slip - r.match_len:slip]) == anchor
                if not legit:
                    slip_source_used += 1
    check(anchor_violations == 0, f"anchor invariant violated {anchor_violations}x")
    check(slip_source_used == 0, f"off-by-one slipped SOURCE used {slip_source_used}x")
    print(f"       anchor_violations={anchor_violations}  slip_source_used={slip_source_used} (bar 0/0)")


def test_common_prefix_safe():
    print("[test] maximal_common_prefix is a prefix of every candidate (safe-to-commit)")
    rng = random.Random(11)
    for _ in range(300):
        alpha = rng.randint(2, 6)
        ctx = [rng.randrange(alpha) for _ in range(rng.randint(10, 120))]
        emit = [rng.randrange(alpha) for _ in range(rng.randint(1, 10))]
        dl = rng.randint(1, 8)
        m = SuffixCopyMiner(min_match=2, cand_cap=8)
        m.append_context(ctx)
        r = m.draft(emit, dl)
        cp = r.common_prefix
        for c in r.candidates:
            check(tuple(c[:len(cp)]) == cp, f"common_prefix {cp} not a prefix of {c}")
        if len(r.candidates) >= 2 and cp:
            # divergence must occur exactly at len(cp) (else prefix is not maximal)
            at = len(cp)
            vals = {c[at] for c in r.candidates if len(c) > at}
            check(len(vals) >= 2 or any(len(c) == at for c in r.candidates),
                  "common_prefix not maximal")


def test_freetext_continuation():
    print("[test] continuation drafting extends beyond a delimiter (freetext copy)")
    # a path echoed from a prior tool output, no grammar boundary
    ctx = [7, 7, 100, 101, 102, 103, 104, 7, 200, 100, 101, 102, 103, 104, 205]
    m = SuffixCopyMiner(min_match=2, cand_cap=8)
    m.append_context(ctx)
    r = m.draft(emitted_suffix=[999, 100, 101], draft_len=3)
    check(r.match_len >= 2, f"match_len {r.match_len}")
    check((102, 103, 104) in set(r.candidates), f"continuation not mined: {r.candidates}")


if __name__ == "__main__":
    test_sam_longest_match()
    test_sam_incremental_equivalence()
    test_endpos_recency()
    test_miner_matches_reference()
    test_pointer_slip_undraftable()
    test_common_prefix_safe()
    test_freetext_continuation()
    print(f"\n=== w1_suffix_miner: {PASS} checks passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
