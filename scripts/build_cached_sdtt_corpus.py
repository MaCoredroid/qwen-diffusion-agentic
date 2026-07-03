#!/usr/bin/env python3
"""Build cached-SDTT targets from audited-correct final-teacher rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import flare_two_stream_noisy_logits, resolve_token_ids  # noqa: E402
from rl_multiturn_grpo_pilot import value_free_token_indices  # noqa: E402
from rl_multiturn_tool_env import EVAL_BATTERY_PATHS, MultiTurnToolRLEnv, write_json, write_jsonl  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "data/cached_sdtt_v2_teacher_probe"
DEFAULT_INPUT = ROOT / "data/rl_multiturn_v2_public_pool/episodes.jsonl"
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_TEACHER = ROOT / "runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model"
DEFAULT_TOKENIZER = ROOT / "models/qwen3.5-9b-fastdllm-b1000-vllm-bf16"
DEFAULT_CHAT_TEMPLATE = Path("/home/mark/shared/lumoFlyWheel_codex_fork/docker/chat_templates/qwen3-openai-codex.jinja")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--teacher-adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--eval-battery-paths", type=Path, nargs="*", default=EVAL_BATTERY_PATHS)
    parser.add_argument("--episode-limit", type=int, default=240)
    parser.add_argument("--target-clean-turns", type=int, default=160)
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--live-tool-json-topk", type=int, default=128)
    parser.add_argument("--max-seq-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--teacher-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--no-merge-adapter", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def env_args(args: argparse.Namespace, rollout_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_jsonl=args.input_jsonl,
        out_dir=rollout_dir,
        eval_battery_paths=args.eval_battery_paths,
        episode_limit=args.episode_limit,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        prompt_tokenizer_path=args.prompt_tokenizer_path,
        tokenizer_path=args.tokenizer_path,
        chat_template_path=args.chat_template_path,
        base_model=args.base_model,
        adapter=args.teacher_adapter,
        no_merge_adapter=args.no_merge_adapter,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        top_p=args.top_p,
        temperature=args.temperature,
        live_tool_json_topk=args.live_tool_json_topk,
        seed=args.seed,
    )


def trim_for_training(
    prompt_ids: list[int],
    assistant_ids: list[int],
    *,
    max_prompt_tokens: int,
    max_seq_tokens: int,
) -> tuple[list[int], list[int], int]:
    if max_prompt_tokens > 0 and len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    room = max_seq_tokens - len(prompt_ids)
    if room <= 0:
        return [], [], 0
    if len(assistant_ids) > room:
        assistant_ids = assistant_ids[:room]
    return prompt_ids, assistant_ids, len(prompt_ids)


def grouped_value_positions(tokenizer, assistant: str, prompt_len: int, assistant_len: int) -> list[list[int]]:
    indices = [idx for idx in value_free_token_indices(tokenizer, assistant) if idx < assistant_len]
    if not indices:
        return []
    groups: list[list[int]] = []
    current: list[int] = []
    previous = None
    for idx in indices:
        if previous is None or idx == previous + 1:
            current.append(prompt_len + idx)
        else:
            groups.append(current)
            current = [prompt_len + idx]
        previous = idx
    if current:
        groups.append(current)
    return groups


@torch.inference_mode()
def cached_target_for_turn(env: MultiTurnToolRLEnv, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    tokenizer = env.tokenizer
    mask_id, _, _ = resolve_token_ids(env.model, tokenizer)
    prompt = str(row.get("prompt") or "")
    assistant = str(row.get("assistant") or "")
    if not prompt or not assistant:
        return None
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(assistant, add_special_tokens=False).input_ids
    prompt_ids, assistant_ids, prompt_len = trim_for_training(
        prompt_ids,
        assistant_ids,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_seq_tokens=int(args.max_seq_tokens),
    )
    if not prompt_ids or not assistant_ids:
        return None
    groups = grouped_value_positions(tokenizer, assistant, prompt_len, len(assistant_ids))
    if not groups:
        return None
    clean_ids = prompt_ids + assistant_ids
    student_mask_positions = sorted({pos for group in groups for pos in group if pos < len(clean_ids)})
    if not student_mask_positions:
        return None
    student_noisy = list(clean_ids)
    for pos in student_mask_positions:
        student_noisy[pos] = int(mask_id)

    teacher_context = list(student_noisy)
    remaining_by_group = [list(group) for group in groups]
    committed_positions: list[int] = []
    for _ in range(max(0, int(args.teacher_steps))):
        for group in remaining_by_group:
            while group and group[0] in committed_positions:
                group.pop(0)
            if not group:
                continue
            pos = group.pop(0)
            teacher_context[pos] = clean_ids[pos]
            committed_positions.append(pos)
    target_positions = sorted(
        pos
        for group in remaining_by_group
        for pos in group
        if pos < len(clean_ids) and teacher_context[pos] == int(mask_id)
    )
    if not target_positions:
        return None

    clean = torch.tensor([clean_ids], dtype=torch.long, device="cuda")
    teacher_noisy = torch.tensor([teacher_context], dtype=torch.long, device="cuda")
    logits = flare_two_stream_noisy_logits(
        env.model,
        clean,
        teacher_noisy,
        block_size=int(args.block_size),
        mask_id=int(mask_id),
    )[:1]
    shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1).float()
    pos_tensor = torch.tensor(target_positions, dtype=torch.long, device="cuda")
    row_logits = shifted[0].index_select(0, pos_tensor)
    row_logits[:, int(mask_id)] = -torch.inf
    topk = min(int(args.top_k), row_logits.shape[-1])
    values, indices = torch.topk(torch.log_softmax(row_logits, dim=-1), k=topk, dim=-1)

    targets = []
    top_ids = indices.detach().cpu().tolist()
    top_logprobs = values.detach().cpu().tolist()
    for pos, ids, logprobs in zip(target_positions, top_ids, top_logprobs):
        targets.append(
            {
                "pos": int(pos),
                "gold_id": int(clean_ids[pos]),
                "top_ids": [int(token_id) for token_id in ids],
                "top_logprobs": [round(float(item), 6) for item in logprobs],
            }
        )
    return {
        "id": row.get("id"),
        "episode_id": row.get("episode_id"),
        "episode_idx": row.get("episode_idx"),
        "turn_idx": row.get("turn_idx"),
        "prompt_tokens": int(prompt_len),
        "assistant_tokens": int(len(assistant_ids)),
        "input_ids": [int(token_id) for token_id in clean_ids],
        "student_noisy_ids": [int(token_id) for token_id in student_noisy],
        "teacher_context_ids": [int(token_id) for token_id in teacher_context],
        "student_mask_positions": [int(pos) for pos in student_mask_positions],
        "teacher_committed_positions": [int(pos) for pos in committed_positions],
        "targets": targets,
        "target_token_count": len(targets),
        "student_mask_token_count": len(student_mask_positions),
        "teacher_steps": int(args.teacher_steps),
        "top_k": int(topk),
        "teacher_rollout_policy": "teacher-forced leftmost gold value token per span per step from audited-correct x0",
        "reverse_kl_caveat": "training consumes sparse top-k targets with reverse-KL over the cached support",
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    totals = manifest["totals"]
    lines = [
        "# Cached-SDTT V2 Teacher Corpus",
        "",
        "Teacher: final v2 adapter. Source rollouts: diffusion careful + live Qwen-native grammar. Kept turns are audit-clean exact-args only.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Episodes rolled out | {totals.get('episodes_rolled_out', 0)} |",
        f"| Turns seen | {totals.get('turns_seen', 0)} |",
        f"| Exact/audit-clean turns | {totals.get('exact_audit_clean_turns', 0)} |",
        f"| Cached records | {totals.get('cached_records', 0)} |",
        f"| Cached target tokens | {totals.get('cached_target_tokens', 0)} |",
        f"| Rejected targetless exact turns | {totals.get('rejected_targetless_exact_turns', 0)} |",
        "",
        "Configuration: `teacher_steps=2`, `top_k=256`, teacher-forced leftmost gold value-token commits per span, sparse top-k reverse-KL caveat.",
        "",
        f"Git HEAD: `{manifest['git_head']}`",
        f"Output JSONL: `{manifest['records_jsonl']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FASTDLLM_FLARE_GDN_ROUTE", "route_i")
    os.environ.setdefault("FASTDLLM_FLARE_TWO_STREAM", "1")
    os.environ.setdefault("FLARE_TWO_STREAM", "1")
    torch.manual_seed(int(args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = args.out_dir / "rollouts"
    env = MultiTurnToolRLEnv(env_args(args, rollout_dir))

    all_turns_path = args.out_dir / "teacher_rollout_turns.jsonl"
    all_rewards_path = args.out_dir / "teacher_rollout_rewards.jsonl"
    records_path = args.out_dir / "cached_sdtt_records.jsonl"
    all_turns_path.write_text("", encoding="utf-8")
    all_rewards_path.write_text("", encoding="utf-8")
    records_path.write_text("", encoding="utf-8")

    totals = Counter()
    started = time.time()
    cached_records = []
    for episode in env.episodes:
        if totals["cached_records"] >= int(args.target_clean_turns):
            break
        rollout = env.rollout_episode(episode)
        totals["episodes_rolled_out"] += 1
        write_jsonl(all_turns_path, []) if False else None
        with all_turns_path.open("a", encoding="utf-8") as handle:
            for row in rollout["turns"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with all_rewards_path.open("a", encoding="utf-8") as handle:
            for row in rollout["rewards"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for turn, reward in zip(rollout["turns"], rollout["rewards"]):
            totals["turns_seen"] += 1
            exact_clean = bool(reward.get("exact_args")) and bool(reward.get("audit_clean"))
            no_value_projection = int(reward.get("projected_value_tokens_exact") or 0) == 0
            if not exact_clean or not no_value_projection:
                totals["rejected_inexact_or_contaminated_turns"] += 1
                continue
            totals["exact_audit_clean_turns"] += 1
            record = cached_target_for_turn(env, turn, args)
            if record is None:
                totals["rejected_targetless_exact_turns"] += 1
                continue
            cached_records.append(record)
            totals["cached_records"] += 1
            totals["cached_target_tokens"] += int(record["target_token_count"])
            totals["student_mask_tokens"] += int(record["student_mask_token_count"])
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if totals["cached_records"] >= int(args.target_clean_turns):
                break

    manifest = {
        "status": "ok" if totals["cached_records"] > 0 else "empty",
        "input_jsonl": str(args.input_jsonl),
        "base_model": str(args.base_model),
        "teacher_adapter": str(args.teacher_adapter),
        "tokenizer_path": str(args.tokenizer_path),
        "chat_template_path": str(args.chat_template_path),
        "chat_template_sha256": sha256_file(args.chat_template_path),
        "records_jsonl": str(records_path),
        "teacher_rollout_turns_jsonl": str(all_turns_path),
        "teacher_rollout_rewards_jsonl": str(all_rewards_path),
        "filtered_public_episodes": str(rollout_dir / "filtered_public_episodes.jsonl"),
        "leak_filter_manifest": str(rollout_dir / "leak_filter_manifest.json"),
        "config": {
            "teacher_steps": int(args.teacher_steps),
            "top_k": int(args.top_k),
            "target_clean_turns": int(args.target_clean_turns),
            "max_seq_tokens": int(args.max_seq_tokens),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "block_size": int(args.block_size),
            "reverse_kl_caveat": True,
            "quality_rl_v5": "held",
        },
        "totals": dict(totals),
        "elapsed_seconds": time.time() - started,
        "git_head": git_head(),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_report(args.out_dir / "report.md", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if totals["cached_records"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
