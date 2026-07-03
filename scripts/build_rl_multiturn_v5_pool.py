#!/usr/bin/env python3
"""Build the RL-v5 mixed pool.

v5 uses the v2 adapter as policy base, so the pool intentionally restores an
easy-anchor fraction while adding fresh public episodes not used in v2/v3/v4.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_flare_broaden_public_eval import row_user_text, split_tool_call_blocks, user_fingerprint  # noqa: E402
from build_rl_multiturn_v2_pool import (  # noqa: E402
    DEFAULT_EVAL_BATTERY,
    DEFAULT_MATCHED20_MANIFEST,
    eval_reference,
    public_eval_hash,
    selected_overlap_counts,
)
from rl_multiturn_tool_env import episode_fingerprint, read_jsonl, write_json, write_jsonl  # noqa: E402
from build_flare_broaden_public_eval import sha256_json  # noqa: E402


DEFAULT_HARD = ROOT / "data/rl_multiturn_v3_frontier_pool/episodes.jsonl"
DEFAULT_EASY = ROOT / "data/rl_multiturn_v3_frontier_pool/dropped_solved4.jsonl"
DEFAULT_CANDIDATES = ROOT / "data/rl_multiturn_v5_mixed_pool/candidate_extended_480.jsonl"
DEFAULT_OUT = ROOT / "data/rl_multiturn_v5_mixed_pool/episodes.jsonl"
DEFAULT_MANIFEST = ROOT / "data/rl_multiturn_v5_mixed_pool/manifest.json"
DEFAULT_USED = [
    ROOT / "data/rl_multiturn_v2_public_pool/episodes.jsonl",
    ROOT / "data/rl_multiturn_v3_frontier_pool/episodes.jsonl",
    ROOT / "data/rl_multiturn_v4_mixed_pool/episodes.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-jsonl", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--easy-jsonl", type=Path, default=DEFAULT_EASY)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fresh-count", type=int, default=60)
    parser.add_argument("--easy-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--used-jsonl", action="append", type=Path, default=list(DEFAULT_USED))
    parser.add_argument("--matched20-manifest", type=Path, default=DEFAULT_MATCHED20_MANIFEST)
    parser.add_argument(
        "--eval-battery-path",
        dest="eval_battery_paths",
        action="append",
        type=Path,
        default=list(DEFAULT_EVAL_BATTERY),
    )
    return parser.parse_args()


def unwrap_easy(row: dict[str, Any]) -> dict[str, Any]:
    if "row" not in row:
        return copy.deepcopy(row)
    item = copy.deepcopy(row["row"])
    if row.get("difficulty_filter") is not None:
        item["rl_v5_easy_anchor_filter"] = row["difficulty_filter"]
    return item


def row_keys(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("id") or ""),
        "fingerprint": episode_fingerprint(row),
        "public_eval_hash": public_eval_hash(row),
        "user_fingerprint": user_fingerprint(row_user_text(row)),
    }


def make_used_sets(paths: list[Path]) -> dict[str, set[str]]:
    used = {"id": set(), "fingerprint": set(), "public_eval_hash": set(), "user_fingerprint": set()}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            item = unwrap_easy(row)
            keys = row_keys(item)
            for name, value in keys.items():
                if value:
                    used[name].add(value)
    return used


def is_used(row: dict[str, Any], used: dict[str, set[str]]) -> bool:
    keys = row_keys(row)
    return any(value and value in used[name] for name, value in keys.items())


def select_fresh(candidates: list[dict[str, Any]], used: dict[str, set[str]], count: int) -> list[dict[str, Any]]:
    selected = []
    seen = {"id": set(), "fingerprint": set(), "public_eval_hash": set(), "user_fingerprint": set()}
    for row in candidates:
        if is_used(row, used):
            continue
        keys = row_keys(row)
        if any(keys[name] and keys[name] in seen[name] for name in seen):
            continue
        item = copy.deepcopy(row)
        item["rl_v5_pool_role"] = "fresh_public_not_used_v2_v3_v4"
        item["rl_v5_fresh_exclusion"] = {
            "not_in_v2_v3_v4_by": ["id", "episode_fingerprint", "public_eval_hash", "user_fingerprint"]
        }
        selected.append(item)
        for name, value in keys.items():
            if value:
                seen[name].add(value)
        if len(selected) >= int(count):
            break
    return selected


def select_easy(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    items = [unwrap_easy(row) for row in rows]
    rng.shuffle(items)
    selected = []
    seen_fingerprints = set()
    for row in items:
        fp = episode_fingerprint(row)
        if fp in seen_fingerprints:
            continue
        item = copy.deepcopy(row)
        item["rl_v5_pool_role"] = "easy_anchor_solved4"
        selected.append(item)
        seen_fingerprints.add(fp)
        if len(selected) >= int(count):
            break
    return selected


def interleave_easy(base_rows: list[dict[str, Any]], easy_rows: list[dict[str, Any]], easy_fraction: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hard_idx = 0
    easy_idx = 0
    total = len(base_rows) + len(easy_rows)
    for _ in range(total):
        target_easy = round((len(out) + 1) * float(easy_fraction))
        should_add_easy = easy_idx < len(easy_rows) and easy_idx < target_easy
        if should_add_easy or hard_idx >= len(base_rows):
            out.append(easy_rows[easy_idx])
            easy_idx += 1
        else:
            out.append(base_rows[hard_idx])
            hard_idx += 1
    return out


def duplicate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {}
    for name in ["id", "fingerprint", "public_eval_hash", "user_fingerprint"]:
        values = [row_keys(row)[name] for row in rows if row_keys(row)[name]]
        counts[name] = len(values) - len(set(values))
    return counts


def row_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "role": row.get("rl_v5_pool_role"),
            "source_family": row.get("source_family") or row.get("source"),
            "source_dataset": row.get("source_dataset"),
            "source_license": row.get("source_license"),
            "turns": len(split_tool_call_blocks(row.get("gold_assistant") or "")),
            "public_eval_hash": public_eval_hash(row),
            "episode_fingerprint": episode_fingerprint(row),
            "user_fingerprint": user_fingerprint(row_user_text(row)),
        }
        for row in rows
    ]


def main() -> int:
    args = parse_args()
    hard_rows = []
    for row in read_jsonl(args.hard_jsonl):
        item = copy.deepcopy(row)
        item["rl_v5_pool_role"] = item.get("rl_v3_pool_role") or "remaining_frontier_or_prior_fresh"
        hard_rows.append(item)
    used = make_used_sets(args.used_jsonl)
    fresh_rows = select_fresh(read_jsonl(args.candidate_jsonl), used, int(args.fresh_count))
    hard_plus_fresh = hard_rows + fresh_rows
    if not fresh_rows:
        raise SystemExit("no fresh public episodes remained after v2/v3/v4 exclusion")
    easy_needed = round((float(args.easy_fraction) / max(1e-9, 1.0 - float(args.easy_fraction))) * len(hard_plus_fresh))
    easy_rows = select_easy(read_jsonl(args.easy_jsonl), easy_needed, int(args.seed))
    selected = interleave_easy(hard_plus_fresh, easy_rows, float(args.easy_fraction))
    eval_ref = eval_reference(args.eval_battery_paths, args.matched20_manifest)
    overlaps = selected_overlap_counts(selected, eval_ref)
    dupes = duplicate_counts(selected)
    if any(overlaps.values()):
        raise SystemExit(f"selected rows overlap frozen eval battery: {overlaps}")
    if any(dupes.values()):
        raise SystemExit(f"selected rows contain duplicates: {dupes}")
    write_jsonl(args.out_jsonl, selected)
    roles = Counter(row.get("rl_v5_pool_role") or "unknown" for row in selected)
    source_families = Counter(row.get("source_family") or row.get("source") or "unknown" for row in selected)
    turn_counts = Counter(str(len(split_tool_call_blocks(row.get("gold_assistant") or ""))) for row in selected)
    group_easy_hist = Counter()
    for idx in range(0, len(selected), 4):
        group = selected[idx : idx + 4]
        group_easy_hist[str(sum(1 for row in group if row.get("rl_v5_pool_role") == "easy_anchor_solved4"))] += 1
    manifest_rows = row_manifest(selected)
    manifest = {
        "out_jsonl": str(args.out_jsonl),
        "records": len(selected),
        "turns": sum(item["turns"] for item in manifest_rows),
        "episode_set_hash": sha256_json(manifest_rows),
        "seed": int(args.seed),
        "easy_fraction_target": float(args.easy_fraction),
        "easy_fraction_actual": roles["easy_anchor_solved4"] / len(selected) if selected else 0.0,
        "fresh_requested": int(args.fresh_count),
        "fresh_selected": len(fresh_rows),
        "easy_selected": len(easy_rows),
        "hard_prior_selected": len(hard_rows),
        "role_counts": dict(sorted(roles.items())),
        "source_family_counts": dict(sorted(source_families.items())),
        "turn_count_histogram": dict(sorted(turn_counts.items())),
        "group_size4_easy_count_histogram": dict(sorted(group_easy_hist.items())),
        "duplicate_counts_selected": dupes,
        "frozen_eval_battery": {
            "paths": [str(path) for path in args.eval_battery_paths],
            "selected_overlap_counts": overlaps,
            "matched20_manifest": str(args.matched20_manifest),
        },
        "inputs": {
            "hard_jsonl": str(args.hard_jsonl),
            "easy_jsonl": str(args.easy_jsonl),
            "candidate_jsonl": str(args.candidate_jsonl),
            "used_jsonl": [str(path) for path in args.used_jsonl],
        },
        "selection_rule": (
            "all v3 frontier/prior-fresh hard rows plus fresh candidates excluded from v2/v3/v4 by "
            "id/fingerprint/public hash/user hash; deterministic solved-easy anchors interleaved to target ~35%"
        ),
        "rows": manifest_rows,
    }
    write_json(args.manifest_json, manifest)
    print(
        json.dumps(
            {
                "records": manifest["records"],
                "turns": manifest["turns"],
                "episode_set_hash": manifest["episode_set_hash"],
                "role_counts": manifest["role_counts"],
                "easy_fraction_actual": manifest["easy_fraction_actual"],
                "selected_overlap_counts": overlaps,
                "duplicate_counts_selected": dupes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
