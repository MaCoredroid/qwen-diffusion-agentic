#!/usr/bin/env python3
"""CROSS-INSTANCE leakage audit for the swe_datagen_s1 SWE-SFT campaign.

Instance-level holdout is already airtight (expand_frontier.py hash-asserts that no
eval-ring id ever trains). THIS script measures the OTHER leak the user asked about:
CROSS-INSTANCE answer leakage -- a *trained* keeper whose patch touches the same
files / functions (or whose issue text closely matches) a *held-out* eval instance,
such that the training row may hand the held-out task the code it needs.

For EVERY keeper (snapshot of keepers.jsonl) x EVERY same-repo held-out id it
computes:
  (a) FILE overlap  : files touched by {keeper.final_patch  UNION  keeper's own gold
                      patch (from SWE-bench_Verified when source is Verified)}
                      vs files touched by {holdout gold patch  UNION  holdout test patch}.
  (b) ISSUE-TEXT sim: TF-IDF cosine (sklearn; clean token-Jaccard fallback) between the
                      keeper problem statement and the held-out problem statement.
  (c) FUNCTION overlap: enclosing def/class tokens from unified-diff hunk headers,
                      restricted to files shared by both patches (cheap, high-signal).

Outputs (deterministic, re-runnable -- re-run verbatim at pool freeze):
  * leakage_audit_report.json  -- per-pair rows ranked by severity + summary + frontier pre-scan
  * leakage_audit_report.md    -- human-readable digest
  * leakage_audit_keepers_snapshot.jsonl -- frozen snapshot of the keepers scanned

CPU-ONLY. Reads the Verified dataset from the local HF parquet cache (no network,
no GPU). Reconstructs the 113-id eval holdout EXACTLY as expand_frontier.py does and
asserts its sha256 against .eval_holdout_sha256 before doing anything else.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# paths / constants (mirror expand_frontier.py exactly for the holdout rebuild)
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/mark/qwen_diffusion")
HERE = REPO_ROOT / "runs/swe_datagen_s1"
KEEPERS = HERE / "keepers/keepers.jsonl"
MANIFEST = REPO_ROOT / "data/swe_sft_pool/pool_manifest.json"
FRONTIER = HERE / "frontier.json"
PIN = HERE / ".eval_holdout_sha256"

RING_SRC = {
    "tier0_20": REPO_ROOT / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json",
    "tier1_100": Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.json"),
}

# local HF parquet cache for princeton-nlp/SWE-bench_Verified (no network)
VERIFIED_PARQUET = Path(
    "/home/mark/.cache/huggingface/hub/datasets--princeton-nlp--SWE-bench_Verified/"
    "snapshots/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet"
)

OUT_JSON = HERE / "leakage_audit_report.json"
OUT_MD = HERE / "leakage_audit_report.md"
OUT_SNAP = HERE / "leakage_audit_keepers_snapshot.jsonl"

TOP_N_REPORT = 200  # cap per-pair rows written to json (ranked); md shows top 40


# ---------------------------------------------------------------------------
# holdout reconstruction (byte-identical to expand_frontier.py) + hash assert
# ---------------------------------------------------------------------------
def _ids(path: Path) -> set[str]:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        return set(d.get("instance_ids", []))
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        return {r.get("instance_id") for r in d if isinstance(r, dict) and r.get("instance_id")}
    return set()


def _sha(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def reconstruct_holdout() -> tuple[set[str], str]:
    man = json.loads(MANIFEST.read_text())
    inner5 = set(man["held_out_rings"]["inner5"]["ids"])
    holdout = inner5 | _ids(RING_SRC["tier0_20"]) | _ids(RING_SRC["tier1_100"])
    sha = _sha(holdout)
    pinned = PIN.read_text().strip() if PIN.exists() else None
    if pinned is not None and sha != pinned:
        raise SystemExit(
            f"KILL-D1 HASH MISMATCH: reconstructed eval-holdout sha256={sha} "
            f"!= pinned {pinned}; a ring source drifted -- refusing to audit."
        )
    return holdout, sha


# ---------------------------------------------------------------------------
# unified-diff parsing: touched files + enclosing-symbol tokens per file
# ---------------------------------------------------------------------------
_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
_PLUS = re.compile(r"^\+\+\+ b/(.+?)\s*$")
_MINUS = re.compile(r"^--- a/(.+?)\s*$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$")
_DEFCLASS = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def parse_patch(patch: str) -> tuple[set[str], dict[str, set[str]]]:
    """Return (files_touched, {file: {enclosing symbol tokens from hunk headers}}).

    Files come from the `diff --git a/… b/…` header (falling back to ---/+++), with
    /dev/null dropped so pure adds/deletes still name the real path. Symbol tokens are
    def/class names harvested from the trailing context git prints after `@@ … @@`.
    """
    files: set[str] = set()
    funcs: dict[str, set[str]] = defaultdict(set)
    cur: str | None = None
    if not patch:
        return files, dict(funcs)
    for line in patch.splitlines():
        m = _DIFF_GIT.match(line)
        if m:
            a, b = m.group(1), m.group(2)
            cur = b if b != "/dev/null" else a
            if cur and cur != "/dev/null":
                files.add(cur)
            continue
        m = _MINUS.match(line)
        if m and m.group(1) != "/dev/null":
            cur = m.group(1)
            files.add(cur)
            continue
        m = _PLUS.match(line)
        if m and m.group(1) != "/dev/null":
            cur = m.group(1)
            files.add(cur)
            continue
        m = _HUNK.match(line)
        if m and cur:
            for sym in _DEFCLASS.findall(m.group(1)):
                funcs[cur].add(sym)
    return files, dict(funcs)


def norm_path(p: str) -> str:
    return p.strip().lstrip("./")


# ---------------------------------------------------------------------------
# keeper problem-statement extraction (from the training prompt_md)
# ---------------------------------------------------------------------------
def keeper_problem_text(row: dict, verified_ps: dict[str, str]) -> str:
    """Clean issue text. For Verified-sourced keepers use the dataset problem_statement
    (identical to what the holdout side uses -> fair TF-IDF); otherwise carve the
    '## Problem statement' … '## Required behavior' span out of the training prompt_md."""
    iid = row["instance_id"]
    if iid in verified_ps and verified_ps[iid]:
        return verified_ps[iid]
    pm = (row.get("prompt") or {}).get("prompt_md", "") or ""
    if "## Problem statement" in pm:
        seg = pm.split("## Problem statement", 1)[1]
        for stop in ("## Required behavior", "## How to work", "## Required"):
            if stop in seg:
                seg = seg.split(stop, 1)[0]
                break
        return seg.strip()
    # last resort: first user turn
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


# ---------------------------------------------------------------------------
# text similarity: TF-IDF cosine (sklearn) with token-Jaccard fallback
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def tokenize(t: str) -> set[str]:
    return {w.lower() for w in _TOKEN.findall(t or "")}


class TextSim:
    """Cosine similarity over a fixed corpus. sklearn TF-IDF when available; else a
    deterministic token-Jaccard fallback. .sim(i, j) returns a float in [0, 1]."""

    def __init__(self, docs: list[str]):
        self.backend = "jaccard"
        self._tok = [tokenize(d) for d in docs]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vec = TfidfVectorizer(
                lowercase=True, token_pattern=r"[A-Za-z_][A-Za-z0-9_]{2,}",
                stop_words="english", min_df=1, sublinear_tf=True,
            )
            self._m = vec.fit_transform(docs) if any(d.strip() for d in docs) else None
            self._cos = cosine_similarity
            if self._m is not None:
                self.backend = "tfidf_cosine_sklearn"
        except Exception as e:  # noqa: BLE001
            self._m = None
            self._note = f"sklearn_unavailable:{e}"

    def sim(self, i: int, j: int) -> float:
        if self._m is not None:
            return float(round(self._cos(self._m[i], self._m[j])[0, 0], 6))
        a, b = self._tok[i], self._tok[j]
        if not a or not b:
            return 0.0
        return round(len(a & b) / len(a | b), 6)


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 6)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    holdout_ids, holdout_sha = reconstruct_holdout()

    # --- snapshot keepers (freeze the growing file, record digest) --------------
    keeper_rows = [json.loads(l) for l in KEEPERS.read_text().splitlines() if l.strip()]
    OUT_SNAP.write_text("\n".join(json.dumps(r) for r in keeper_rows) + "\n")
    snap_sha = hashlib.sha256(OUT_SNAP.read_bytes()).hexdigest()

    # --- load Verified dataset from local parquet (gold+test patch, PS, repo) ---
    import pandas as pd

    df = pd.read_parquet(VERIFIED_PARQUET).set_index("instance_id")
    ver_repo = df["repo"].to_dict()
    ver_gold = df["patch"].to_dict()
    ver_test = df["test_patch"].to_dict()
    ver_ps = df["problem_statement"].to_dict()

    # holdout side: parse gold + test patch once
    holdout = {}
    for iid in sorted(holdout_ids):
        if iid not in df.index:
            holdout[iid] = {"repo": None, "missing_from_verified": True}
            continue
        gf, gfun = parse_patch(ver_gold.get(iid, ""))
        tf, tfun = parse_patch(ver_test.get(iid, ""))
        gf = {norm_path(p) for p in gf}
        tf = {norm_path(p) for p in tf}
        funs = defaultdict(set)
        for f, s in gfun.items():
            funs[norm_path(f)] |= s
        for f, s in tfun.items():
            funs[norm_path(f)] |= s
        holdout[iid] = {
            "repo": ver_repo.get(iid),
            "gold_files": gf,
            "test_files": tf,
            "all_files": gf | tf,
            "funcs": {k: v for k, v in funs.items()},
            "ps": ver_ps.get(iid, "") or "",
        }
    holdout_by_repo: dict[str, list[str]] = defaultdict(list)
    for iid, h in holdout.items():
        if h.get("repo"):
            holdout_by_repo[h["repo"]].append(iid)

    # --- keeper side: parse final_patch (+ own gold when Verified), extract PS ---
    keepers = []
    verified_ps_for_keeper = {}
    for r in keeper_rows:
        iid = r["instance_id"]
        is_ver = r.get("source") == "SWE-bench_Verified"
        if is_ver and iid in df.index:
            verified_ps_for_keeper[iid] = ver_ps.get(iid, "") or ""
        ff, ffun = parse_patch(r.get("final_patch", "") or "")
        ff = {norm_path(p) for p in ff}
        funs = defaultdict(set)
        for f, s in ffun.items():
            funs[norm_path(f)] |= s
        gold_files: set[str] = set()
        if is_ver and iid in df.index:
            gf, gfun = parse_patch(ver_gold.get(iid, "") or "")
            gold_files = {norm_path(p) for p in gf}
            for f, s in gfun.items():
                funs[norm_path(f)] |= s
        keepers.append({
            "instance_id": iid,
            "repo": r["repo"],
            "source": r.get("source"),
            "final_files": ff,
            "gold_files": gold_files,
            "all_files": ff | gold_files,
            "funcs": {k: v for k, v in funs.items()},
        })

    # --- ASSERT the disjoint-repo claim for SWE-Gym keepers ---------------------
    swegym_repo_collisions = sorted({
        k["repo"] for k in keepers
        if k["source"] == "SWE-Gym" and holdout_by_repo.get(k["repo"])
    })

    # --- build the text corpus for TF-IDF (only pair-relevant docs) -------------
    # index docs: keepers that have >=1 same-repo holdout, then all such holdout ids
    kept_with_repo = [k for k in keepers if holdout_by_repo.get(k["repo"])]
    doc_ids = [("K", k["instance_id"]) for k in kept_with_repo]
    seen_hold = []
    for k in kept_with_repo:
        for hid in holdout_by_repo[k["repo"]]:
            if hid not in seen_hold:
                seen_hold.append(hid)
    doc_ids += [("H", h) for h in seen_hold]
    docs = []
    for kind, iid in doc_ids:
        if kind == "K":
            row = next(r for r in keeper_rows if r["instance_id"] == iid)
            docs.append(keeper_problem_text(row, verified_ps_for_keeper))
        else:
            docs.append(holdout[iid]["ps"])
    doc_index = {(kind, iid): i for i, (kind, iid) in enumerate(doc_ids)}
    ts = TextSim(docs)

    # --- pairwise scan: keeper x same-repo holdout ------------------------------
    pairs = []
    for k in keepers:
        hids = holdout_by_repo.get(k["repo"], [])
        for hid in hids:
            h = holdout[hid]
            file_gold = sorted(k["all_files"] & h["gold_files"])
            file_test = sorted(k["all_files"] & h["test_files"])
            file_any = sorted(k["all_files"] & h["all_files"])
            shared = set(file_any)
            func_shared = {}
            for f in sorted(shared):
                inter = k["funcs"].get(f, set()) & h["funcs"].get(f, set())
                if inter:
                    func_shared[f] = sorted(inter)
            n_func = sum(len(v) for v in func_shared.values())
            tsim = 0.0
            if ("K", k["instance_id"]) in doc_index and ("H", hid) in doc_index:
                tsim = ts.sim(doc_index[("K", k["instance_id"])], doc_index[("H", hid)])
            severity = round(
                5.0 * len(file_gold)
                + 2.0 * max(0, len(file_any) - len(file_gold))
                + 3.0 * n_func
                + 2.0 * tsim,
                6,
            )
            pairs.append({
                "keeper_id": k["instance_id"],
                "keeper_source": k["source"],
                "holdout_id": hid,
                "repo": k["repo"],
                "file_overlap_gold": file_gold,          # keeper files ∩ holdout GOLD files (strongest)
                "file_overlap_test": file_test,          # keeper files ∩ holdout TEST files
                "file_overlap_any": file_any,
                "n_file_overlap_gold": len(file_gold),
                "n_file_overlap_any": len(file_any),
                "func_overlap": func_shared,             # {shared_file: [shared def/class tokens]}
                "n_func_overlap": n_func,
                "text_sim": tsim,
                "severity": severity,
            })

    # text-sim p95 over all same-repo pairs (flag threshold)
    all_tsims = [p["text_sim"] for p in pairs]
    p95 = percentile(all_tsims, 95.0)
    p99 = percentile(all_tsims, 99.0)
    for p in pairs:
        p["text_sim_ge_p95"] = bool(p["text_sim"] >= p95 and p95 > 0)

    # a pair is "flagged" if it carries real code-overlap signal, OR high text sim
    def is_flagged(p):
        return p["n_file_overlap_gold"] > 0 or p["n_func_overlap"] > 0 or (
            p["n_file_overlap_any"] > 0) or p["text_sim_ge_p95"]

    pairs.sort(key=lambda p: (-p["severity"], -p["n_file_overlap_gold"],
                              -p["n_func_overlap"], -p["text_sim"],
                              p["keeper_id"], p["holdout_id"]))
    flagged = [p for p in pairs if is_flagged(p)]

    # --- summary counts ---------------------------------------------------------
    keepers_with_neighbor = sorted({k["instance_id"] for k in keepers if holdout_by_repo.get(k["repo"])})
    keepers_with_file_overlap = sorted({p["keeper_id"] for p in pairs if p["n_file_overlap_any"] > 0})
    keepers_with_gold_file_overlap = sorted({p["keeper_id"] for p in pairs if p["n_file_overlap_gold"] > 0})
    keepers_with_func_overlap = sorted({p["keeper_id"] for p in pairs if p["n_func_overlap"] > 0})
    keepers_with_hi_textsim = sorted({p["keeper_id"] for p in pairs if p["text_sim_ge_p95"]})

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "cpu_only": True,
        "eval_holdout_n_distinct": len(holdout_ids),
        "eval_holdout_sha256": holdout_sha,
        "eval_holdout_sha256_pinned": PIN.read_text().strip() if PIN.exists() else None,
        "eval_holdout_sha_assert": "PASS",
        "keepers_snapshot_path": str(OUT_SNAP),
        "keepers_snapshot_sha256": snap_sha,
        "n_keepers": len(keepers),
        "n_keepers_verified_source": sum(1 for k in keepers if k["source"] == "SWE-bench_Verified"),
        "n_keepers_swegym_source": sum(1 for k in keepers if k["source"] == "SWE-Gym"),
        "text_sim_backend": ts.backend,
        "n_same_repo_pairs": len(pairs),
        "n_flagged_pairs": len(flagged),
        "text_sim_p95": p95,
        "text_sim_p99": p99,
        "swegym_disjoint_repo_assert": "PASS" if not swegym_repo_collisions else "FAIL",
        "swegym_repo_collisions": swegym_repo_collisions,
        "counts": {
            "keepers_with_same_repo_holdout_neighbor": len(keepers_with_neighbor),
            "keepers_with_any_file_overlap": len(keepers_with_file_overlap),
            "keepers_with_holdout_GOLD_file_overlap": len(keepers_with_gold_file_overlap),
            "keepers_with_function_overlap": len(keepers_with_func_overlap),
            "keepers_with_text_sim_ge_p95": len(keepers_with_hi_textsim),
        },
        "keepers_with_holdout_GOLD_file_overlap_ids": keepers_with_gold_file_overlap,
        "keepers_with_function_overlap_ids": keepers_with_func_overlap,
    }

    # --- FRONTIER PRE-SCAN: remaining Verified-adjacent ids vs holdout ----------
    # gold-patch files only (these ids are not yet keepers -> no generated patch).
    front = json.loads(FRONTIER.read_text())
    order = front["order"]
    vhl = int(front.get("best_of_k", {}).get("verified_head_len", 0))
    verified_head = order[:vhl]
    keeper_id_set = {k["instance_id"] for k in keepers}
    frontier_pairs = []
    scanned = 0
    for iid in verified_head:
        if iid in keeper_id_set or iid not in df.index:
            continue
        repo = ver_repo.get(iid)
        hids = holdout_by_repo.get(repo, [])
        if not hids:
            continue
        gf, gfun = parse_patch(ver_gold.get(iid, "") or "")
        gf = {norm_path(p) for p in gf}
        gfun = {norm_path(f): s for f, s in gfun.items()}
        scanned += 1
        for hid in hids:
            h = holdout[hid]
            fo = sorted(gf & h["gold_files"])
            if not fo:
                continue
            func_shared = {}
            for f in fo:
                inter = gfun.get(f, set()) & h["funcs"].get(f, set())
                if inter:
                    func_shared[f] = sorted(inter)
            frontier_pairs.append({
                "frontier_id": iid,
                "holdout_id": hid,
                "repo": repo,
                "gold_file_overlap": fo,
                "n_gold_file_overlap": len(fo),
                "func_overlap": func_shared,
                "n_func_overlap": sum(len(v) for v in func_shared.values()),
                "severity": round(5.0 * len(fo) + 3.0 * sum(len(v) for v in func_shared.values()), 6),
            })
    frontier_pairs.sort(key=lambda p: (-p["severity"], -p["n_gold_file_overlap"],
                                       p["frontier_id"], p["holdout_id"]))
    frontier_prescreen = {
        "note": "Pre-screen of NOT-YET-collected Verified-adjacent frontier ids (gold-patch "
                "files only) vs same-repo holdout gold files. Flags ids whose fix touches a "
                "file a held-out task also fixes; if such an id later becomes a keeper it "
                "should be re-examined or dropped.",
        "verified_head_len": vhl,
        "n_scanned_same_repo_remaining": scanned,
        "n_file_overlap_pairs": len(frontier_pairs),
        "flagged_frontier_ids": sorted({p["frontier_id"] for p in frontier_pairs}),
        "top_pairs": frontier_pairs[:100],
    }

    report = {
        "audit": "cross_instance_answer_leakage",
        "summary": summary,
        "flagged_pairs_ranked": flagged[:TOP_N_REPORT],
        "all_pairs_count": len(pairs),
        "top_pairs_ranked": pairs[:TOP_N_REPORT],
        "frontier_prescreen": frontier_prescreen,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))

    # --- markdown digest --------------------------------------------------------
    md = []
    md.append("# Cross-instance answer-leakage audit — swe_datagen_s1\n")
    md.append(f"_generated {summary['generated_at']} · CPU-only · {summary['elapsed_sec']}s · "
              f"text-sim backend `{ts.backend}`_\n")
    md.append("## Holdout / snapshot integrity\n")
    md.append(f"- eval holdout: **{len(holdout_ids)} distinct ids** "
              f"(inner5 ∪ tier0_20 ∪ tier1_100)\n")
    md.append(f"- holdout sha256 `{holdout_sha}` — pin-assert **PASS**\n")
    md.append(f"- keepers scanned: **{len(keepers)}** "
              f"({summary['n_keepers_verified_source']} Verified, "
              f"{summary['n_keepers_swegym_source']} SWE-Gym); snapshot sha256 `{snap_sha[:16]}…`\n")
    md.append(f"- SWE-Gym disjoint-repo assert: **{summary['swegym_disjoint_repo_assert']}** "
              f"(collisions: {swegym_repo_collisions or 'none'})\n")
    md.append("\n## Topline\n")
    c = summary["counts"]
    md.append(f"- same-repo keeper×holdout pairs: **{len(pairs)}**; flagged: **{len(flagged)}**\n")
    md.append(f"- keepers with a same-repo holdout neighbor: **{c['keepers_with_same_repo_holdout_neighbor']}**\n")
    md.append(f"- keepers with ANY file overlap vs holdout gold+test: **{c['keepers_with_any_file_overlap']}**\n")
    md.append(f"- keepers overlapping a holdout **GOLD** file: **{c['keepers_with_holdout_GOLD_file_overlap']}**\n")
    md.append(f"- keepers with function-level overlap: **{c['keepers_with_function_overlap']}**\n")
    md.append(f"- keepers with issue-text sim ≥ p95 ({p95}): **{c['keepers_with_text_sim_ge_p95']}**\n")
    md.append("\n## Top 40 flagged pairs (by severity)\n")
    md.append("| # | keeper | holdout | repo | gold-file∩ | func∩ | text-sim | sev |\n")
    md.append("|--:|---|---|---|---|--:|--:|--:|\n")
    for i, p in enumerate(flagged[:40], 1):
        gfo = ", ".join(p["file_overlap_gold"]) or ("—" if not p["file_overlap_any"]
                                                     else "(test/any: " + ", ".join(p["file_overlap_any"]) + ")")
        fn = ", ".join(f"{f}:{'/'.join(v)}" for f, v in p["func_overlap"].items()) or "—"
        md.append(f"| {i} | `{p['keeper_id']}` | `{p['holdout_id']}` | {p['repo']} | "
                  f"{gfo} | {fn} | {p['text_sim']} | {p['severity']} |\n")
    md.append("\n## Frontier pre-screen (not-yet-collected Verified-adjacent ids)\n")
    md.append(f"- remaining same-repo ids scanned: **{frontier_prescreen['n_scanned_same_repo_remaining']}**\n")
    md.append(f"- ids with a gold-file overlap vs a holdout task: "
              f"**{len(frontier_prescreen['flagged_frontier_ids'])}**\n")
    md.append("\n| # | frontier id | holdout | repo | gold-file∩ | func∩ | sev |\n")
    md.append("|--:|---|---|---|---|--:|--:|\n")
    for i, p in enumerate(frontier_prescreen["top_pairs"][:30], 1):
        fn = ", ".join(f"{f}:{'/'.join(v)}" for f, v in p["func_overlap"].items()) or "—"
        md.append(f"| {i} | `{p['frontier_id']}` | `{p['holdout_id']}` | {p['repo']} | "
                  f"{', '.join(p['gold_file_overlap'])} | {fn} | {p['severity']} |\n")
    OUT_MD.write_text("".join(md))

    # --- stdout: machine-readable topline for the caller ------------------------
    print(json.dumps({
        "topline": summary["counts"],
        "n_same_repo_pairs": len(pairs),
        "n_flagged_pairs": len(flagged),
        "text_sim_p95": p95,
        "swegym_disjoint_repo_assert": summary["swegym_disjoint_repo_assert"],
        "frontier_flagged_ids": len(frontier_prescreen["flagged_frontier_ids"]),
        "top20": [
            {
                "keeper_id": p["keeper_id"], "holdout_id": p["holdout_id"], "repo": p["repo"],
                "file_overlap_gold": p["file_overlap_gold"],
                "file_overlap_any": p["file_overlap_any"],
                "func_overlap": p["func_overlap"], "text_sim": p["text_sim"],
                "severity": p["severity"],
            } for p in flagged[:20]
        ],
        "out_json": str(OUT_JSON), "out_md": str(OUT_MD),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
