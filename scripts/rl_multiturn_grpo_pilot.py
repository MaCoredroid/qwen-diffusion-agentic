#!/usr/bin/env python3
"""Small audited multi-turn diffu-GRPO pilot.

This is deliberately a pilot, not the full RL trainer. It starts from the
Run-1 copy-grounded checkpoint, samples grouped diffusion rollouts with the
live Qwen-native tool grammar, scores them with the audited ToolRL-style
reward, then applies a group-relative policy update by replaying generated
assistant tokens through the trainable LoRA under raw full-vocab logprob.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import resolve_single_token_ids, resolve_token_ids  # noqa: E402
from eval_flare_multiturn_percall_waves import build_episodes  # noqa: E402
from eval_flare_northstar_matched import DEFAULT_AR_MODEL, DEFAULT_CHAT_TEMPLATE, DEFAULT_DIFFUSION_BASE, load_chat_template  # noqa: E402
from rl_multiturn_tool_env import (  # noqa: E402
    DEFAULT_PUBLIC_INPUT,
    EVAL_BATTERY_PATHS,
    MultiTurnToolRLEnv,
    configure_env,
    filter_eval_battery_rows,
    make_env_args,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_RUN1_ADAPTER = ROOT / "runs/flare_redesign_run1_copy_grounded_qwen35_9b"
DEFAULT_OUT_DIR = ROOT / "runs/rl_multiturn_grpo_pilot/run1_smoke_g2_step2"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def shift_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def grpo_advantages(rewards: list[float]) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device="cuda")
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return (values - values.mean()) / std.clamp_min(1e-6)


def configure_cuda_env() -> None:
    configure_env()
    os.environ.setdefault("FASTDLLM_GDN_KERNEL", "torch")


def load_trainable_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(args.gpu_index)},
    )
    base.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    elif hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=bool(args.gradient_checkpointing))
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    model.config.use_cache = False
    model.train()
    return model, tokenizer


class TrainableMultiTurnToolRLEnv(MultiTurnToolRLEnv):
    def __init__(self, args: argparse.Namespace):
        configure_cuda_env()
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
        self.model, self.tokenizer = load_trainable_model(args)
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


def trainable_parameter_count(model) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)


PARAMETER_RE = re.compile(r"<parameter=[^>\n]+>\s*\n?(.*?)\n?</parameter>", re.DOTALL)


def value_free_token_indices(tokenizer, assistant: str) -> list[int]:
    """Return assistant-token indices inside raw parameter-value spans.

    The live grammar determines XML structure, tool names, parameter names, and
    closing tags. Those positions are masked out of GRPO replay. Parameter value
    text is the free/raw part of the policy and is the only replay target here.
    """
    spans: list[tuple[int, int]] = []
    for match in PARAMETER_RE.finditer(assistant):
        value_start, value_end = match.span(1)
        while value_start < value_end and assistant[value_start] in "\r\n\t ":
            value_start += 1
        while value_end > value_start and assistant[value_end - 1] in "\r\n\t ":
            value_end -= 1
        if value_end > value_start:
            spans.append((value_start, value_end))
    if not spans:
        return []
    encoded = tokenizer(
        assistant,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    selected: list[int] = []
    for idx, (start, end) in enumerate(encoded.offset_mapping):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            selected.append(idx)
    return selected


def replay_examples(tokenizer, rollout: dict, advantage: float, args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = []
    for row in rollout["turns"]:
        prompt = str(row.get("prompt") or "")
        assistant = str(row.get("assistant") or "")
        if not prompt or not assistant:
            continue
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        assistant_ids = tokenizer(assistant, add_special_tokens=False).input_ids
        if args.replay_max_prompt_tokens and len(prompt_ids) > args.replay_max_prompt_tokens:
            prompt_ids = prompt_ids[-int(args.replay_max_prompt_tokens) :]
        max_assistant = int(args.replay_max_seq_tokens) - len(prompt_ids)
        if max_assistant <= 0:
            continue
        assistant_ids = assistant_ids[:max_assistant]
        policy_token_indices = [idx for idx in value_free_token_indices(tokenizer, assistant) if idx < len(assistant_ids)]
        if not assistant_ids:
            continue
        if not policy_token_indices:
            continue
        examples.append(
            {
                "prompt_ids": prompt_ids,
                "assistant_ids": assistant_ids,
                "policy_token_indices": policy_token_indices,
                "advantage": float(advantage),
                "episode_id": row.get("episode_id"),
                "turn_idx": row.get("turn_idx"),
                "exact_args": bool(row.get("exact_arguments")),
                "reward": (rollout.get("rewards") or [{}])[int(row.get("turn_idx") or 0)].get("reward"),
                "assistant_tokens": len(assistant_ids),
                "grammar_forced_tokens_masked": len(assistant_ids) - len(policy_token_indices),
            }
        )
    return examples


def backward_replay_loss(model, tokenizer, examples: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(len(item["policy_token_indices"]) for item in examples)
    if total_tokens <= 0:
        return {"policy_tokens": 0, "grammar_forced_tokens_masked": 0, "loss": 0.0, "mean_logprob": None}
    loss_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    logprob_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    masked_tokens = 0
    for item in examples:
        ids = item["prompt_ids"] + item["assistant_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        output = model(input_ids=input_ids, use_cache=False)
        logits = shift_logits(output.logits)
        start = len(item["prompt_ids"])
        selected = torch.tensor(item["policy_token_indices"], dtype=torch.long, device="cuda")
        assistant_ids = torch.tensor(item["assistant_ids"], dtype=torch.long, device="cuda")
        targets = assistant_ids.index_select(0, selected)
        row_logits = logits[0, start : start + len(item["assistant_ids"]), :].index_select(0, selected)
        logprobs = torch.log_softmax(row_logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        unscaled = -float(item["advantage"]) * logprobs.sum()
        (unscaled / float(total_tokens)).backward()
        loss_sum = loss_sum + unscaled.detach().float()
        logprob_sum = logprob_sum + logprobs.detach().float().sum()
        masked_tokens += int(item.get("grammar_forced_tokens_masked") or 0)
        del output, logits, input_ids, selected, assistant_ids, targets, row_logits, logprobs
    return {
        "policy_tokens": int(total_tokens),
        "grammar_forced_tokens_masked": int(masked_tokens),
        "loss": float((loss_sum / float(total_tokens)).detach().cpu().item()),
        "mean_logprob": float((logprob_sum / float(total_tokens)).detach().cpu().item()),
    }


def summarize_rollout(rollout: dict, group_idx: int, reward: float, advantage: float) -> dict[str, Any]:
    rewards = rollout.get("rewards") or []
    return {
        "group_idx": group_idx,
        "episode_id": rollout.get("episode_id"),
        "turns": len(rewards),
        "reward": float(reward),
        "advantage": float(advantage),
        "exact_turns": sum(int(bool(item.get("exact_args"))) for item in rewards),
        "episode_exact_all_turns": bool(rollout.get("episode_exact_all_turns")),
        "audit_clean": bool(rollout.get("episode_audit_clean")),
        "turn_rewards": [float(item.get("reward") or 0.0) for item in rewards],
    }


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    configure_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multi-turn GRPO pilot")
    torch.cuda.set_device(int(args.gpu_index))
    torch.manual_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    env = TrainableMultiTurnToolRLEnv(args)
    trainable, total = trainable_parameter_count(env.model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in env.model.parameters() if parameter.requires_grad),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    config = {
        "args": jsonable(vars(args)),
        "trainable_params": trainable,
        "total_params": total,
        "warm_start": str(args.adapter),
        "policy": "diffusion careful + live Qwen-native grammar, audited reward, GRPO group advantages",
        "logprob_replay": "raw full-vocab replay over parameter-value/free assistant tokens only",
        "grammar_forced_policy_masking": "XML structure, tool names, parameter names, and tag whitespace are masked out of policy loss",
        "stop_rule_context": "SFT warm-start abandoned after repaired gate failed GSM8K retention",
        "run1_anchor": "Run-1 copy-grounded checkpoint previously validated at GSM8K 0.75 and matched-20 careful 34/63",
        "grouping": "mixed adjacent public episodes" if args.mixed_episode_groups else "same prompt per GRPO group",
        "leak_filter_manifest": env.leak_manifest,
        "episodes": len(env.episodes),
    }
    write_json(args.out_dir / "config.json", config)
    print("[config] " + json.dumps(config, sort_keys=True), flush=True)

    step_rows = []
    started_all = time.perf_counter()
    for step in range(1, int(args.max_steps) + 1):
        base_episode = env.episodes[(step - 1) % len(env.episodes)]
        rollouts = []
        rewards = []
        step_start = time.perf_counter()
        for group_idx in range(int(args.group_size)):
            episode = (
                env.episodes[(step - 1 + group_idx) % len(env.episodes)]
                if args.mixed_episode_groups
                else base_episode
            )
            torch.manual_seed(int(args.seed) + step * 1009 + group_idx)
            env.model.eval()
            rollout = env.rollout_episode(episode)
            reward = float(rollout.get("episode_reward_mean") or 0.0)
            rollouts.append(rollout)
            rewards.append(reward)
            write_jsonl(args.out_dir / f"step_{step:04d}_rollout_{group_idx:02d}.turns.jsonl", rollout["turns"])
            write_jsonl(args.out_dir / f"step_{step:04d}_rollout_{group_idx:02d}.rewards.jsonl", rollout["rewards"])
        advantages = grpo_advantages(rewards)
        examples = []
        rollout_summaries = []
        for group_idx, rollout in enumerate(rollouts):
            adv = float(advantages[group_idx].detach().cpu().item())
            rollout_summaries.append(summarize_rollout(rollout, group_idx, rewards[group_idx], adv))
            if abs(adv) > 1e-8:
                examples.extend(replay_examples(env.tokenizer, rollout, adv, args))

        update_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        env.model.train()
        if examples:
            loss_metrics = backward_replay_loss(env.model, env.tokenizer, examples)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in env.model.parameters() if parameter.requires_grad],
                float(args.max_grad_norm),
            )
            optimizer.step()
            grad_norm_value = float(grad_norm.detach().float().cpu().item())
        else:
            loss_metrics = {"policy_tokens": 0, "loss": 0.0, "mean_logprob": None}
            grad_norm_value = 0.0
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_start
        row = {
            "step": step,
            "episode_id": base_episode["id"],
            "group_episode_ids": [rollout.get("episode_id") for rollout in rollouts],
            "turns": len(base_episode["gold_blocks"]),
            "rewards": rewards,
            "reward_mean": sum(rewards) / max(1, len(rewards)),
            "advantages": [float(item) for item in advantages.detach().cpu().tolist()],
            "zero_advantage": bool(torch.count_nonzero(advantages).item() == 0),
            "rollouts": rollout_summaries,
            "examples": len(examples),
            "grammar_forced_tokens_masked": sum(int(item.get("grammar_forced_tokens_masked") or 0) for item in examples),
            "grad_norm": grad_norm_value,
            "rollout_seconds": update_start - step_start,
            "update_seconds": update_seconds,
            "step_seconds": time.perf_counter() - step_start,
            **loss_metrics,
        }
        step_rows.append(row)
        append_jsonl(metrics_path, row)
        print(
            "[train] "
            f"step={step} reward={row['reward_mean']:.3f} "
            f"zero_adv={row['zero_advantage']} tokens={row['policy_tokens']} "
            f"loss={row['loss']:.4g} grad={row['grad_norm']:.3f} "
            f"rollout_s={row['rollout_seconds']:.2f} update_s={row['update_seconds']:.2f}",
            flush=True,
        )

    adapter_out = args.out_dir / "adapter_model"
    env.model.save_pretrained(adapter_out)
    env.tokenizer.save_pretrained(args.out_dir)
    elapsed = time.perf_counter() - started_all
    summary = {
        "status": "pilot_complete",
        "steps": len(step_rows),
        "group_size": int(args.group_size),
        "elapsed_seconds": elapsed,
        "nonzero_advantage_steps": sum(int(not row["zero_advantage"]) for row in step_rows),
        "policy_tokens": sum(int(row["policy_tokens"]) for row in step_rows),
        "grammar_forced_tokens_masked": sum(int(row.get("grammar_forced_tokens_masked") or 0) for row in step_rows),
        "mean_reward": sum(float(row["reward_mean"]) for row in step_rows) / max(1, len(step_rows)),
        "last_step": step_rows[-1] if step_rows else None,
        "adapter_out": str(adapter_out),
        "trainable_params": trainable,
        "warm_start": str(args.adapter),
        "gate_note": "SFT warm-start is abandoned; this pilot starts from Run-1 copy-grounded checkpoint.",
    }
    write_json(args.out_dir / "summary.json", summary)
    report = [
        "# Multi-Turn diffu-GRPO Pilot",
        "",
        f"- Warm start: `{args.adapter}`",
        f"- Steps: `{len(step_rows)}`",
        f"- Group size: `{int(args.group_size)}`",
        f"- Grouping: `{'mixed adjacent public episodes' if args.mixed_episode_groups else 'same prompt'}`",
        f"- Nonzero-advantage steps: `{summary['nonzero_advantage_steps']}/{len(step_rows)}`",
        f"- Policy replay tokens: `{summary['policy_tokens']}`",
        f"- Grammar-forced tokens masked from policy loss: `{summary['grammar_forced_tokens_masked']}`",
        f"- Mean step reward: `{summary['mean_reward']:.4f}`",
        f"- Output adapter: `{adapter_out}`",
        "",
        "Rollouts used diffusion careful decode with live Qwen-native grammar and audited ToolRL-style reward. "
        "The update is a raw full-vocab logprob replay approximation over generated parameter-value/free tokens only; "
        "grammar-forced structure is excluded from the policy loss. This is a plumbing pilot, not a promoted training result.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_PUBLIC_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=1)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--prompt-tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_AR_MODEL)
    parser.add_argument("--chat-template-path", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_DIFFUSION_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_RUN1_ADAPTER)
    parser.add_argument("--no-merge-adapter", action="store_true", default=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--live-tool-json-topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-max-prompt-tokens", type=int, default=768)
    parser.add_argument("--replay-max-seq-tokens", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument(
        "--mixed-episode-groups",
        action="store_true",
        help="Use adjacent public episodes inside each group when same-prompt rollouts are reward-identical.",
    )
    parser.add_argument(
        "--eval-battery-path",
        dest="eval_battery_paths",
        action="append",
        type=Path,
        default=list(EVAL_BATTERY_PATHS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
