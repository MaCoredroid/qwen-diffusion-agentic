#!/usr/bin/env python3
"""Anchor-gate scorer (KILL-T1): paired matched-20 tool-call exact_args, post-SFT vs pre-SFT.

Reads two AR-guided turns.jsonl (pre-SFT base and post-SFT candidate), pairs turn-by-turn
by (episode_id, turn_idx) with gold_sha256 identity assertion, and reports:
  - raw exact_arguments / valid_tool_call / episode_exact for each arm
  - paired McNemar b (pre-right,post-wrong), c (post-right,pre-wrong), net-loss b-c, two-sided exact-binomial p
  - PASS bar (design 2.5 / KILL-T1): McNemar net-loss NOT significant (p>=0.05) AND raw_post >= anchor - 3
"""
import argparse, json, math, sys
from pathlib import Path


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return {(r["episode_id"], r["turn_idx"]): r for r in rows}


def exact(r):
    return bool(r.get("exact_arguments"))


def valid(r):
    return bool(r.get("valid_tool_call"))


def two_sided_binom_p(b, c):
    # exact binomial test of net discordance, H0: p=0.5 on the b+c discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: sum of probabilities of outcomes as or more extreme
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return p


def episode_exact_counts(turns):
    eps = {}
    for (eid, tidx), r in turns.items():
        eps.setdefault(eid, []).append(exact(r))
    return sum(1 for v in eps.values() if all(v)), len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, type=Path, help="pre-SFT base turns.jsonl")
    ap.add_argument("--post", required=True, type=Path, help="post-SFT candidate turns.jsonl")
    ap.add_argument("--anchor", type=int, default=50, help="banked pre-SFT exact_args anchor (default 50/63)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pre, post = load(args.pre), load(args.post)
    keys = sorted(set(pre) & set(post))
    only_pre = sorted(set(pre) - set(post))
    only_post = sorted(set(post) - set(pre))

    gold_mismatch = [k for k in keys if pre[k].get("gold_sha256") != post[k].get("gold_sha256")]

    pre_exact = sum(exact(pre[k]) for k in keys)
    post_exact = sum(exact(post[k]) for k in keys)
    pre_valid = sum(valid(pre[k]) for k in keys)
    post_valid = sum(valid(post[k]) for k in keys)

    b = sum(1 for k in keys if exact(pre[k]) and not exact(post[k]))  # pre-right, post-wrong (erosion)
    c = sum(1 for k in keys if exact(post[k]) and not exact(pre[k]))  # post-right, pre-wrong (gain)
    net_loss = b - c
    p = two_sided_binom_p(b, c)

    pre_ep_ex, pre_ep_n = episode_exact_counts(pre)
    post_ep_ex, post_ep_n = episode_exact_counts(post)

    raw_ok = post_exact >= (args.anchor - 3)
    mcnemar_ok = p >= 0.05  # net-loss not significant
    passed = raw_ok and mcnemar_ok

    # discordant turn ids for the erosion profile
    eroded = [k for k in keys if exact(pre[k]) and not exact(post[k])]
    gained = [k for k in keys if exact(post[k]) and not exact(pre[k])]

    res = {
        "n_paired_turns": len(keys),
        "only_in_pre": only_pre,
        "only_in_post": only_post,
        "gold_sha256_mismatch_count": len(gold_mismatch),
        "gold_sha256_mismatch_keys": gold_mismatch[:20],
        "anchor_exact_args": args.anchor,
        "pre_sft": {
            "exact_args": f"{pre_exact}/{len(keys)}",
            "valid_tool_call": f"{pre_valid}/{len(keys)}",
            "episode_exact": f"{pre_ep_ex}/{pre_ep_n}",
        },
        "post_sft": {
            "exact_args": f"{post_exact}/{len(keys)}",
            "valid_tool_call": f"{post_valid}/{len(keys)}",
            "episode_exact": f"{post_ep_ex}/{post_ep_n}",
        },
        "mcnemar": {
            "b_pre_right_post_wrong": b,
            "c_post_right_pre_wrong": c,
            "net_loss_b_minus_c": net_loss,
            "p_two_sided_exact": round(p, 6),
        },
        "erosion_profile": {
            "eroded_turns_pre_right_post_wrong": [f"{e}#t{t}" for (e, t) in eroded],
            "gained_turns_post_right_pre_wrong": [f"{e}#t{t}" for (e, t) in gained],
        },
        "gates": {
            "raw_post_ge_anchor_minus_3": {"value": post_exact, "bar": args.anchor - 3, "PASS": raw_ok},
            "mcnemar_netloss_not_significant_p_ge_0.05": {"p": round(p, 6), "PASS": mcnemar_ok},
        },
        "VERDICT": "PASS" if passed else "FAIL",
    }
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out:
        args.out.write_text(txt)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
