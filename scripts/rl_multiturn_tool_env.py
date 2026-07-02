#!/usr/bin/env python3
"""Multi-turn tool-call RL environment skeleton.

This is the RL prep scaffold only: public episode loading, label-free careful
diffusion rollouts with live structure grammar, audited reward scoring, and a
1-episode smoke driver. It intentionally does not run optimizer steps or long
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_value_projection_tokens import audit_rows  # noqa: E402
from eval_fastdllm_toolcall_cases import full_context_sample, load_model, resolve_single_token_ids, resolve_token_ids  # noqa: E402
from eval_flare_multiturn_percall_waves import build_episodes, make_gen_args, synthetic_tool_result  # noqa: E402
from eval_flare_northstar_matched import (  # noqa: E402
    DEFAULT_AR_MODEL,
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_DIFFUSION_ADAPTER,
    DEFAULT_DIFFUSION_BASE,
    ASSISTANT_GENERATION_PROMPT,
    decode_text,
    load_chat_template,
    next_turn_user_message,
    render_matched_prompt,
    row_from_generation,
    tool_response_suffix,
    trim_scored_assistant,
)
from eval_toolcall_jsonl import extract_tool_calls, normalize_call_for_compare, score_tool_calls, tool_schema_by_name  # noqa: E402
from flare_hf_cache import FlarePrefixCache  # noqa: E402


DEFAULT_PUBLIC_INPUT = ROOT / "data/toolcall_eval_native/public_multicall_qwen_native_smoke.jsonl"
DEFAULT_OUT_DIR = ROOT / "runs/rl_multiturn_prep/smoke_1episode"
EVAL_BATTERY_PATHS = [
    ROOT / "data/toolcall_eval_native/flare_scaleup_native_58.jsonl",
    ROOT / "data/toolcall_eval_native/flare_nevertrain_bfcl_apibank.jsonl",
    ROOT / "data/toolcall_eval_native/flare_broaden_public_toolace60.jsonl",
]


def configure_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ["FASTDLLM_FLARE_TWO_STREAM"] = "1"
    os.environ["FLARE_TWO_STREAM"] = "1"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def episode_fingerprint(row: dict) -> str:
    return sha256_json(
        {
            "prompt_messages": row.get("prompt_messages") or row.get("messages") or [],
            "tools": row.get("tools") or [],
            "gold_assistant": row.get("gold_assistant") or "",
        }
    )


def eval_battery_fingerprints(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("id"):
                ids.add(str(row["id"]))
            fingerprints.add(episode_fingerprint(row))
    return ids, fingerprints


def filter_eval_battery_rows(input_jsonl: Path, rows: list[dict], eval_paths: list[Path]) -> tuple[list[dict], dict]:
    eval_ids, eval_fps = eval_battery_fingerprints(eval_paths)
    kept = []
    rejected = []
    for row in rows:
        row_id = str(row.get("id") or "")
        row_fp = episode_fingerprint(row)
        reason = None
        if row_id and row_id in eval_ids:
            reason = "eval_id_overlap"
        elif row_fp in eval_fps:
            reason = "eval_fingerprint_overlap"
        if reason:
            rejected.append({"id": row_id, "source": row.get("source"), "reason": reason})
        else:
            kept.append(row)
    manifest = {
        "input_jsonl": str(input_jsonl),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(rejected),
        "eval_battery_paths": [str(path) for path in eval_paths],
        "rejected_preview": rejected[:20],
    }
    return kept, manifest


@dataclass
class RewardBreakdown:
    reward: float
    exact_args: bool
    format_reward: float
    schema_reward: float
    tool_name_reward: float
    arg_name_reward: float
    value_reward: float
    audit_clean: bool
    audit_mode: str
    projected_value_tokens_exact: int
    metrics: dict[str, Any]


def argument_partial_credit(pred_calls: list[dict], gold_calls: list[dict], schemas: dict[str, Any]) -> tuple[float, float]:
    if not gold_calls:
        return 0.0, 0.0
    aligned = list(zip(pred_calls, gold_calls))
    if not aligned:
        return 0.0, 0.0
    arg_name_hits = 0
    arg_name_total = 0
    value_hits = 0
    value_total = 0
    for pred, gold in aligned:
        pred_n = normalize_call_for_compare(pred, schemas)
        gold_n = normalize_call_for_compare(gold, schemas)
        pred_args = pred_n.get("arguments") if isinstance(pred_n.get("arguments"), dict) else {}
        gold_args = gold_n.get("arguments") if isinstance(gold_n.get("arguments"), dict) else {}
        for key, gold_value in gold_args.items():
            arg_name_total += 1
            if key in pred_args:
                arg_name_hits += 1
                value_total += 1
                value_hits += int(pred_args.get(key) == gold_value)
            else:
                value_total += 1
    return arg_name_hits / max(1, arg_name_total), value_hits / max(1, value_total)


def audited_reward(tokenizer, row: dict, tools: list[dict], gold_block: str) -> RewardBreakdown:
    metrics = score_tool_calls(row.get("assistant") or "", tools, gold_block)
    audit_totals, _ = audit_rows(tokenizer, [row])
    audit_clean = bool(audit_totals.get("zero_projected_value_tokens_verified"))
    schemas = tool_schema_by_name(tools)
    pred_calls, _ = extract_tool_calls(row.get("assistant") or "")
    gold_calls, _ = extract_tool_calls(gold_block)
    arg_name_score, value_score = argument_partial_credit(pred_calls, gold_calls, schemas)
    name_score = 1.0 if metrics.get("exact_tool_sequence") else 0.0
    if not name_score and gold_calls:
        pred_names = [call.get("name") for call in pred_calls]
        gold_names = [call.get("name") for call in gold_calls]
        overlap = sum((Counter(pred_names) & Counter(gold_names)).values())
        name_score = overlap / max(1, len(gold_names))
    format_reward = 1.0 if metrics.get("valid_tool_call") else 0.0
    schema_reward = 1.0 if metrics.get("all_schema_valid") and metrics.get("all_required_args_present") else 0.0
    exact = bool(metrics.get("exact_arguments"))
    if exact:
        reward = 1.0
    else:
        reward = (
            0.10 * format_reward
            + 0.15 * schema_reward
            + 0.20 * name_score
            + 0.20 * arg_name_score
            + 0.35 * value_score
        )
        reward = min(0.99, reward)
    if not audit_clean:
        reward = 0.0
    return RewardBreakdown(
        reward=float(reward),
        exact_args=exact,
        format_reward=float(format_reward),
        schema_reward=float(schema_reward),
        tool_name_reward=float(name_score),
        arg_name_reward=float(arg_name_score),
        value_reward=float(value_score),
        audit_clean=audit_clean,
        audit_mode=str(audit_totals.get("verification_mode")),
        projected_value_tokens_exact=int(audit_totals.get("projected_value_tokens_exact") or 0),
        metrics=metrics,
    )


def make_env_args(args: argparse.Namespace, filtered_jsonl: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_jsonl=filtered_jsonl,
        episode_limit=args.episode_limit,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        prompt_tokenizer_path=args.prompt_tokenizer_path,
        chat_template_path=args.chat_template_path,
        ar_model_path=args.prompt_tokenizer_path,
        ar_served_model="unused",
        ar_base_url="unused",
        base_model=args.base_model,
        adapter=args.adapter,
        no_merge_adapter=args.no_merge_adapter,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        diffusion_condition="baseline_careful_live_grammar",
        diffusion_structural_only=False,
        out_dir=args.out_dir,
        seed=args.seed,
    )


class MultiTurnToolRLEnv:
    def __init__(self, args: argparse.Namespace):
        configure_env()
        self.args = args
        self.out_dir = args.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rows = read_jsonl(args.input_jsonl)
        kept, self.leak_manifest = filter_eval_battery_rows(args.input_jsonl, rows, args.eval_battery_paths)
        if not kept:
            raise SystemExit("no public training rows remain after eval-battery filtering")
        self.filtered_jsonl = self.out_dir / "filtered_public_episodes.jsonl"
        write_jsonl(self.filtered_jsonl, kept)
        write_json(self.out_dir / "leak_filter_manifest.json", self.leak_manifest)
        self.eval_args = make_env_args(args, self.filtered_jsonl)
        self.prompt_tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_path), trust_remote_code=True)
        self.chat_template = load_chat_template(args.chat_template_path)
        self.episodes = build_episodes(self.eval_args)
        self.model, self.tokenizer = load_model(
            args.base_model,
            args.adapter if args.adapter and args.adapter.exists() else None,
            merge_adapter=not args.no_merge_adapter,
            tokenizer_path=args.tokenizer_path,
        )
        self.model.eval()
        if hasattr(self.model, "config"):
            setattr(self.model.config, "bd_size", int(args.block_size))
        mask_id, stop_token_id, base_stop_token_ids = resolve_token_ids(self.model, self.tokenizer)
        tool_close_ids = self.tokenizer("</tool_call>", add_special_tokens=False).input_ids
        self.stop_token_ids = list(dict.fromkeys([int(item) for item in base_stop_token_ids + tool_close_ids]))
        self.mask_id = int(mask_id)
        self.stop_token_id = int(stop_token_id)
        self.argument_boundary_token_ids = resolve_single_token_ids(
            self.tokenizer, ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]
        )
        self.argument_newline_token_ids = resolve_single_token_ids(self.tokenizer, ["\n", "\n\n"])

    def gen_args(self, prefix_cache: FlarePrefixCache) -> SimpleNamespace:
        gen_args = make_gen_args(
            self.eval_args,
            condition="baseline_careful",
            prefix_cache=prefix_cache,
            mask_id=self.mask_id,
            stop_token_id=self.stop_token_id,
            stop_token_ids=self.stop_token_ids,
            argument_boundary_token_ids=self.argument_boundary_token_ids,
            argument_newline_token_ids=self.argument_newline_token_ids,
        )
        gen_args.parallel_commit_threshold = None
        gen_args.parallel_commit_kinds = set()
        gen_args.live_tool_json_grammar = True
        gen_args.live_tool_json_topk = int(self.args.live_tool_json_topk)
        gen_args.record_projected_token_positions = True
        gen_args.tool_prefix_guard_mode = "qwen_native"
        return gen_args

    def rollout_episode(self, episode: dict) -> dict:
        prefix_cache = FlarePrefixCache()
        gen_args = self.gen_args(prefix_cache)
        prompt = render_matched_prompt(
            self.tokenizer,
            [dict(message) for message in episode["prompt_messages"]],
            episode["tools"],
            self.chat_template,
        )
        turn_rows = []
        reward_rows = []
        for turn_idx, gold_block in enumerate(episode["gold_blocks"]):
            prompt_input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            gen_args.max_new_tokens = int(gen_args.max_new_tokens_cap)
            previous_live_tool_schemas = getattr(gen_args, "_live_tool_schemas", None)
            gen_args._live_tool_schemas = tool_schema_by_name(episode["tools"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            try:
                with torch.no_grad():
                    generated = full_context_sample(
                        self.model,
                        prompt_input_ids,
                        self.tokenizer,
                        gen_args,
                        sampler_schedule=None,
                        original_len_override=prompt_input_ids.shape[1],
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                if previous_live_tool_schemas is None:
                    try:
                        delattr(gen_args, "_live_tool_schemas")
                    except AttributeError:
                        pass
                else:
                    gen_args._live_tool_schemas = previous_live_tool_schemas
            new_ids = generated[prompt_input_ids.shape[1] :]
            history_text = decode_text(self.tokenizer, new_ids)
            assistant_text = trim_scored_assistant(history_text)
            row = row_from_generation(
                backend="diffusion_careful_live_grammar_rl_env",
                episode=episode,
                turn_idx=turn_idx,
                prompt=prompt,
                tools=episode["tools"],
                gold_block=gold_block,
                assistant_text=assistant_text,
                prompt_tokens=int(prompt_input_ids.shape[1]),
                generated_tokens=int((new_ids != gen_args.mask_id).sum().item()),
                turn_wall_seconds=time.time() - start,
                schedule_build_seconds=0.0,
                backend_meta={
                    "sampler_schedule_events": getattr(gen_args, "_last_sampler_schedule_events", {}),
                    "flare_cache_stats": getattr(gen_args, "_last_flare_cache_stats", {}),
                    "flare_prefix_cache_stats": getattr(gen_args, "_last_flare_prefix_cache_stats", {}),
                    "flare_timing_stats": getattr(gen_args, "_last_flare_timing_stats", {}),
                    "max_new_tokens": gen_args.max_new_tokens,
                    "sampler_schedule_used": False,
                    "policy": "diffusion_careful_live_grammar_label_free_structure_values_raw",
                    "assistant_generation_prompt": ASSISTANT_GENERATION_PROMPT,
                },
            )
            row["generated_token_ids"] = [int(token_id) for token_id in new_ids.detach().cpu().tolist()]
            reward = audited_reward(self.tokenizer, row, episode["tools"], gold_block)
            reward_payload = {
                "episode_id": episode["id"],
                "turn_idx": turn_idx,
                "reward": reward.reward,
                "exact_args": reward.exact_args,
                "format_reward": reward.format_reward,
                "schema_reward": reward.schema_reward,
                "tool_name_reward": reward.tool_name_reward,
                "arg_name_reward": reward.arg_name_reward,
                "value_reward": reward.value_reward,
                "audit_clean": reward.audit_clean,
                "audit_mode": reward.audit_mode,
                "projected_value_tokens_exact": reward.projected_value_tokens_exact,
                "valid_tool_json": bool(reward.metrics.get("valid_tool_call")),
                "all_schema_valid": bool(reward.metrics.get("all_schema_valid")),
                "exact_tool_sequence": bool(reward.metrics.get("exact_tool_sequence")),
                "called_names": reward.metrics.get("called_names") or [],
            }
            turn_rows.append(row)
            reward_rows.append(reward_payload)
            next_user = next_turn_user_message(episode, turn_idx + 1)
            prompt = prompt + history_text + tool_response_suffix(row["tool_response_payload"], next_user)
            print(
                f"rollout episode={episode['id']} turn={turn_idx + 1}/{len(episode['gold_blocks'])} "
                f"reward={reward.reward:.3f} exact={int(reward.exact_args)} audit={int(reward.audit_clean)}",
                flush=True,
            )
        return {
            "episode_id": episode["id"],
            "turns": turn_rows,
            "rewards": reward_rows,
            "episode_reward_mean": sum(item["reward"] for item in reward_rows) / max(1, len(reward_rows)),
            "episode_exact_all_turns": all(bool(item["exact_args"]) for item in reward_rows),
            "episode_audit_clean": all(bool(item["audit_clean"]) for item in reward_rows),
        }


def grpo_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards))
    std = variance ** 0.5
    if std <= 1e-8:
        return [0.0 for _ in rewards]
    return [(reward - mean) / std for reward in rewards]


def write_smoke_outputs(args: argparse.Namespace, rollout: dict, leak_manifest: dict) -> None:
    turn_rows = rollout["turns"]
    reward_rows = rollout["rewards"]
    write_jsonl(args.out_dir / "turns.jsonl", turn_rows)
    write_jsonl(args.out_dir / "rewards.jsonl", reward_rows)
    audit_totals, audit_detail = audit_rows(
        AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True),
        turn_rows,
    )
    write_json(args.out_dir / "projection_value_audit.json", {"totals": audit_totals})
    write_jsonl(args.out_dir / "projection_value_audit.jsonl", audit_detail)
    rewards = [float(item["reward"]) for item in reward_rows]
    live_grammar_token_visits = sum(
        int(((row.get("backend_meta") or {}).get("sampler_schedule_events") or {}).get("live_tool_json_grammar_token_visits") or 0)
        for row in turn_rows
    )
    parallel_commit_forwards = sum(
        int(((row.get("backend_meta") or {}).get("sampler_schedule_events") or {}).get("parallel_commit_denoise_forwards") or 0)
        for row in turn_rows
    )
    summary = {
        "objective": "RL env skeleton smoke; no training/optimizer step",
        "episode_id": rollout["episode_id"],
        "turns": len(turn_rows),
        "exact_args": sum(int(bool(item["exact_args"])) for item in reward_rows),
        "episode_exact_all_turns": bool(rollout["episode_exact_all_turns"]),
        "episode_reward_mean": rollout["episode_reward_mean"],
        "reward_values": rewards,
        "grpo_advantages_single_group": grpo_advantages(rewards),
        "audit_totals": audit_totals,
        "leak_filter_manifest": leak_manifest,
        "policy": {
            "decode": "diffusion-careful",
            "structure_constraint": "live Qwen-native grammar, label-free schema/tool names only",
            "values": "raw model tokens",
            "sampler_schedule": "none; no gold token schedule",
            "waves": "retired/off",
            "parallel_commit": "off",
            "live_grammar_token_visits": live_grammar_token_visits,
            "parallel_commit_forwards": parallel_commit_forwards,
        },
        "reward": {
            "promotion_gate": "exact_args",
            "graded_terms": {
                "format": 0.10,
                "schema_required": 0.15,
                "tool_name_sequence_or_overlap": 0.20,
                "argument_name": 0.20,
                "argument_value": 0.35,
            },
            "audit_gate": "zero projected value tokens required",
        },
        "training_guards": {
            "kl_to_base": "skeleton hook; not run in smoke",
            "retention_guard": "GSM8K >= 0.70 before any promotion",
            "eval_battery": [
                "matched-20",
                "scaleup_native_58",
                "nevertrain BFCL/API-Bank",
            ],
        },
    }
    write_json(args.out_dir / "summary.json", summary)
    report = [
        "# RL Multi-Turn Env Smoke",
        "",
        "No training was run. This smoke exercises public episode loading, eval-battery filtering, careful+live-grammar rollout, audited reward scoring, and GRPO advantage calculation.",
        "",
        f"- Episode: `{rollout['episode_id']}`",
        f"- Turns: {len(turn_rows)}",
        f"- Exact args: {summary['exact_args']}/{len(turn_rows)}",
        f"- Mean reward: {summary['episode_reward_mean']:.3f}",
        f"- Audit: mode `{audit_totals.get('verification_mode')}`, projected_value_tokens_exact={audit_totals.get('projected_value_tokens_exact')}, verified={audit_totals.get('zero_projected_value_tokens_verified')}",
        f"- Live grammar token visits: {live_grammar_token_visits}; parallel-commit forwards: {parallel_commit_forwards}.",
        f"- Leak filter: kept {leak_manifest['kept_rows']}/{leak_manifest['input_rows']} rows; rejected {leak_manifest['rejected_rows']}.",
        "",
        "Reward design follows the taxonomy: format/schema and wrong-value terms dominate direct misses, with exact-args retained as the gate and episode-level accounting available for generated-history compounding.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_PUBLIC_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=1)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DIFFUSION_ADAPTER)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--merge-adapter", dest="no_merge_adapter", action="store_false")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--live-tool-json-topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--eval-battery-path",
        dest="eval_battery_paths",
        action="append",
        type=Path,
        default=list(EVAL_BATTERY_PATHS),
    )
    parser.add_argument("--smoke", action="store_true", help="Run one rollout and write smoke artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    env = MultiTurnToolRLEnv(args)
    manifest = {
        "episodes": len(env.episodes),
        "input_jsonl": str(args.input_jsonl),
        "filtered_jsonl": str(env.filtered_jsonl),
        "leak_filter": env.leak_manifest,
        "policy": "diffusion-careful + live grammar structure, values raw",
        "training": "not run",
    }
    write_json(args.out_dir / "env_manifest.json", manifest)
    if args.smoke:
        rollout = env.rollout_episode(env.episodes[0])
        write_smoke_outputs(args, rollout, env.leak_manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
