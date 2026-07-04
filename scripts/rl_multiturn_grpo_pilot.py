#!/usr/bin/env python3
"""Small audited multi-turn diffu-GRPO pilot.

This is deliberately a pilot, not the full RL trainer. It samples grouped
diffusion rollouts, scores them with the audited ToolRL-style reward, then
applies a group-relative policy update by replaying generated assistant tokens
through the trainable LoRA under raw full-vocab logprob.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_fastdllm_toolcall_cases import resolve_single_token_ids, resolve_token_ids  # noqa: E402
from eval_flare_stage1_ab_diffusion import (  # noqa: E402
    build_gsm8k_prompt,
    full_context_sample_one,
    gsm8k_gold,
    gsm8k_strict,
    normalize_number,
)
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
DEFAULT_GSM8K = ROOT / "data/phaseA_retention/gsm8k_main_test_first20.jsonl"
DEFAULT_GSM8K_FEWSHOT = ROOT / "data/phaseA_retention/gsm8k_main_train_first5.jsonl"


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


def trainable_state_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model, snapshot: dict[str, torch.Tensor]) -> None:
    if not snapshot:
        return
    by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = by_name.get(name)
            if parameter is None:
                continue
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


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
                "policy_value_tokens": len(policy_token_indices),
                "policy_free_tokens": 0,
            }
        )
    return examples


def reference_kl_sum(
    model,
    input_ids: torch.Tensor,
    row_slice: slice,
    selected: torch.Tensor,
    current_row_logits: torch.Tensor,
    current_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    restore_trainable_state(model, reference_state)
    try:
        with torch.no_grad():
            ref_output = model(input_ids=input_ids, use_cache=False)
            ref_logits = shift_logits(ref_output.logits)
            ref_row_logits = ref_logits[0, row_slice, :].index_select(0, selected).float()
    finally:
        restore_trainable_state(model, current_state)
    temp = max(float(temperature), 1e-6)
    current_logprobs = torch.log_softmax(current_row_logits.float() / temp, dim=-1)
    ref_logprobs = torch.log_softmax(ref_row_logits / temp, dim=-1)
    ref_probs = ref_logprobs.exp()
    return (ref_probs * (ref_logprobs - current_logprobs)).sum(dim=-1).sum()


def backward_replay_loss(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    reference_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    total_tokens = sum(len(item["policy_token_indices"]) for item in examples)
    if total_tokens <= 0:
        return {"policy_tokens": 0, "grammar_forced_tokens_masked": 0, "loss": 0.0, "mean_logprob": None}
    loss_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    policy_loss_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    kl_loss_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    logprob_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    masked_tokens = 0
    kl_tokens = 0
    kl_coeff = float(getattr(args, "kl_to_base_coeff", 0.0) or 0.0)
    current_state = trainable_state_snapshot(model) if reference_state and kl_coeff > 0 else {}
    for item in examples:
        ids = item["prompt_ids"] + item["assistant_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        output = model(input_ids=input_ids, use_cache=False)
        logits = shift_logits(output.logits)
        start = len(item["prompt_ids"])
        selected = torch.tensor(item["policy_token_indices"], dtype=torch.long, device="cuda")
        assistant_ids = torch.tensor(item["assistant_ids"], dtype=torch.long, device="cuda")
        targets = assistant_ids.index_select(0, selected)
        row_slice = slice(start, start + len(item["assistant_ids"]))
        row_logits = logits[0, row_slice, :].index_select(0, selected)
        logprobs = torch.log_softmax(row_logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        policy_unscaled = -float(item["advantage"]) * logprobs.sum()
        kl_unscaled = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        if reference_state and kl_coeff > 0:
            kl_unscaled = reference_kl_sum(
                model,
                input_ids,
                row_slice,
                selected,
                row_logits,
                current_state,
                reference_state,
                float(args.kl_to_base_temperature),
            )
            kl_tokens += int(selected.numel())
        unscaled = policy_unscaled + kl_coeff * kl_unscaled
        (unscaled / float(total_tokens)).backward()
        loss_sum = loss_sum + unscaled.detach().float()
        policy_loss_sum = policy_loss_sum + policy_unscaled.detach().float()
        kl_loss_sum = kl_loss_sum + kl_unscaled.detach().float()
        logprob_sum = logprob_sum + logprobs.detach().float().sum()
        masked_tokens += int(item.get("grammar_forced_tokens_masked") or 0)
        del output, logits, input_ids, selected, assistant_ids, targets, row_logits, logprobs
    return {
        "policy_tokens": int(total_tokens),
        "grammar_forced_tokens_masked": int(masked_tokens),
        "loss": float((loss_sum / float(total_tokens)).detach().cpu().item()),
        "policy_loss": float((policy_loss_sum / float(total_tokens)).detach().cpu().item()),
        "kl_to_base_coeff": float(kl_coeff),
        "kl_to_base_tokens": int(kl_tokens),
        "kl_to_base_loss": float((kl_loss_sum / float(max(1, kl_tokens))).detach().cpu().item()) if kl_tokens else 0.0,
        "kl_to_base_loss_scaled_per_policy_token": float(
            ((kl_coeff * kl_loss_sum) / float(total_tokens)).detach().cpu().item()
        )
        if kl_tokens
        else 0.0,
        "mean_logprob": float((logprob_sum / float(total_tokens)).detach().cpu().item()),
    }


def run_retention_probe(env: TrainableMultiTurnToolRLEnv, args: argparse.Namespace, step: int) -> dict[str, Any]:
    gsm_rows = read_jsonl(Path(args.retention_gsm8k_path))[: int(args.retention_probe_limit)]
    fewshot_rows = read_jsonl(Path(args.retention_gsm8k_fewshot_path))[: int(args.retention_gsm8k_fewshot)]
    probe_dir = args.out_dir / f"retention_probe_step_{step:04d}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    gen_args = SimpleNamespace(
        mask_id=int(env.mask_id),
        stop_token_id=int(env.stop_token_id),
        stop_token_ids=list(env.stop_token_ids),
        block_size=int(args.retention_block_size),
        small_block_size=int(args.retention_small_block_size),
        max_new_tokens=int(args.retention_max_new_tokens),
        threshold=float(args.retention_threshold),
        top_p=float(args.retention_top_p),
        temperature=float(args.retention_temperature),
        fresh_generation_blocks=True,
    )
    was_training = bool(env.model.training)
    env.model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for idx, row in enumerate(gsm_rows):
            prompt = build_gsm8k_prompt(env.tokenizer, row, fewshot_rows)
            input_ids = env.tokenizer([prompt], return_tensors="pt").input_ids[0].cpu()
            with torch.no_grad():
                output_ids, sampler_metrics = full_context_sample_one(env.model, input_ids, gen_args)
            new_ids = output_ids[int(input_ids.numel()) :]
            text = env.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            gold = gsm8k_gold(str(row.get("answer") or ""))
            strict_pred = gsm8k_strict(text)
            flex_pred = normalize_number(text)
            rows.append(
                {
                    "task": "gsm8k_quick_retention",
                    "step": step,
                    "idx": row.get("idx", idx),
                    "gold": gold,
                    "strict_pred": strict_pred,
                    "flex_pred": flex_pred,
                    "strict_correct": strict_pred == gold,
                    "flex_correct": flex_pred == gold,
                    "prompt_tokens": int(input_ids.numel()),
                    "generated_tokens": int((new_ids != env.mask_id).sum().item()),
                    "mask_count": int((new_ids == env.mask_id).sum().item()),
                    "generated": text,
                    "sampler": sampler_metrics,
                }
            )
    finally:
        if was_training:
            env.model.train()
    strict_correct = sum(int(item["strict_correct"]) for item in rows)
    flex_correct = sum(int(item["flex_correct"]) for item in rows)
    examples = len(rows)
    summary = {
        "step": step,
        "examples": examples,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / examples if examples else 0.0,
        "flex_correct": flex_correct,
        "flex_accuracy": flex_correct / examples if examples else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "collapse_threshold_flex_accuracy": float(args.retention_collapse_flex_accuracy),
        "early_stop_collapse": bool(
            examples > 0 and (flex_correct / examples) < float(args.retention_collapse_flex_accuracy)
        ),
        "adapter": str(args.adapter),
        "probe_dir": str(probe_dir),
    }
    write_jsonl(probe_dir / "gsm8k_quick_rows.jsonl", rows)
    write_json(probe_dir / "summary.json", summary)
    append_jsonl(args.out_dir / "retention_probes.jsonl", summary)
    return summary


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
    reference_state = trainable_state_snapshot(env.model) if float(args.kl_to_base_coeff) > 0 else None
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
        "policy": (
            "diffusion hybrid-clean + audited reward + GRPO group advantages"
            if args.decode_policy == "hybrid_clean"
            else "diffusion careful + live Qwen-native grammar, audited reward, GRPO group advantages"
        ),
        "decode_policy": str(args.decode_policy),
        "logprob_replay": "raw full-vocab replay over parameter-value/free assistant tokens only",
        "grammar_forced_policy_masking": "XML structure, tool names, parameter names, and tag whitespace are masked out of policy loss",
        "kl_to_base": {
            "enabled": bool(reference_state is not None),
            "coefficient": float(args.kl_to_base_coeff),
            "temperature": float(args.kl_to_base_temperature),
            "reference": "initial trainable LoRA adapter tensor snapshot on CPU",
            "positions": "same parameter-value/free assistant tokens as policy loss",
            "early_stop_window": int(args.kl_early_stop_window),
            "early_stop_mean_threshold": float(args.kl_early_stop_mean_threshold),
        },
        "retention_probe": {
            "every_steps": int(args.retention_probe_every_steps),
            "limit": int(args.retention_probe_limit),
            "gsm8k_path": str(args.retention_gsm8k_path),
            "fewshot_path": str(args.retention_gsm8k_fewshot_path),
            "collapse_flex_accuracy": float(args.retention_collapse_flex_accuracy),
        },
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
            loss_metrics = backward_replay_loss(env.model, env.tokenizer, examples, args, reference_state)
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
            "policy_value_tokens": sum(int(item.get("policy_value_tokens") or 0) for item in examples),
            "policy_free_tokens": sum(int(item.get("policy_free_tokens") or 0) for item in examples),
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
            f"loss={row['loss']:.4g} kl={row.get('kl_to_base_loss', 0.0):.4g} "
            f"grad={row['grad_norm']:.3f} "
            f"rollout_s={row['rollout_seconds']:.2f} update_s={row['update_seconds']:.2f}",
            flush=True,
        )
        kl_window = int(args.kl_early_stop_window)
        kl_threshold = float(args.kl_early_stop_mean_threshold)
        if kl_window > 0 and kl_threshold > 0 and len(step_rows) >= kl_window:
            recent_kl = [float(item.get("kl_to_base_loss") or 0.0) for item in step_rows[-kl_window:]]
            kl_mean = sum(recent_kl) / float(kl_window)
            row["kl_early_stop_window"] = kl_window
            row["kl_early_stop_mean"] = kl_mean
            row["kl_early_stop_mean_threshold"] = kl_threshold
            if kl_mean > kl_threshold:
                row["early_stop_reason"] = "kl_last_window_mean"
                append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "event": "early_stop",
                        "early_stop_reason": row["early_stop_reason"],
                        "kl_window": kl_window,
                        "kl_mean": kl_mean,
                        "kl_mean_threshold": kl_threshold,
                    },
                )
                print(
                    "[early-stop] "
                    f"step={step} kl_last_{kl_window}_mean={kl_mean:.4g} "
                    f"> threshold={kl_threshold:.4g}",
                    flush=True,
                )
                break
        if int(args.retention_probe_every_steps) > 0 and step % int(args.retention_probe_every_steps) == 0:
            probe = run_retention_probe(env, args, step)
            row["retention_probe"] = probe
            if probe["early_stop_collapse"]:
                row["early_stop_reason"] = "retention_probe_collapse"
                append_jsonl(metrics_path, {"step": step, "event": "early_stop", "retention_probe": probe})
                print(
                    "[early-stop] "
                    f"step={step} retention_flex={probe['flex_accuracy']:.3f} "
                    f"< threshold={probe['collapse_threshold_flex_accuracy']:.3f}",
                    flush=True,
                )
                break

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
        "policy_value_tokens": sum(int(row.get("policy_value_tokens") or 0) for row in step_rows),
        "policy_free_tokens": sum(int(row.get("policy_free_tokens") or 0) for row in step_rows),
        "mean_reward": sum(float(row["reward_mean"]) for row in step_rows) / max(1, len(step_rows)),
        "last_step": step_rows[-1] if step_rows else None,
        "early_stopped": bool(step_rows and step_rows[-1].get("early_stop_reason")),
        "early_stop_reason": step_rows[-1].get("early_stop_reason") if step_rows else None,
        "kl_to_base_coeff": float(args.kl_to_base_coeff),
        "kl_early_stop_window": int(args.kl_early_stop_window),
        "kl_early_stop_mean_threshold": float(args.kl_early_stop_mean_threshold),
        "retention_probe_every_steps": int(args.retention_probe_every_steps),
        "adapter_out": str(adapter_out),
        "trainable_params": trainable,
        "warm_start": str(args.adapter),
        "gate_note": f"Pilot starts from configured warm-start adapter: {args.adapter}",
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
        f"- Policy value tokens: `{summary['policy_value_tokens']}`",
        f"- Policy free tokens: `{summary['policy_free_tokens']}`",
        f"- Grammar-forced tokens masked from policy loss: `{summary['grammar_forced_tokens_masked']}`",
        f"- KL-to-base coefficient: `{float(args.kl_to_base_coeff)}`",
        f"- KL early stop: last `{int(args.kl_early_stop_window)}` mean > `{float(args.kl_early_stop_mean_threshold)}`",
        f"- Retention probe cadence: every `{int(args.retention_probe_every_steps)}` steps, limit `{int(args.retention_probe_limit)}`",
        f"- Early stopped: `{summary['early_stopped']}` ({summary['early_stop_reason']})",
        f"- Mean step reward: `{summary['mean_reward']:.4f}`",
        f"- Output adapter: `{adapter_out}`",
        "",
        f"Rollouts used `{args.decode_policy}` decode with audited ToolRL-style reward. "
        "The update is a raw full-vocab logprob replay approximation over generated parameter-value/free tokens only; "
        "grammar-forced structure is excluded from the policy loss.",
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
    parser.add_argument(
        "--decode-policy",
        choices=["careful_live_grammar", "hybrid_clean"],
        default="careful_live_grammar",
    )
    parser.add_argument("--hybrid-grammar-topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--kl-to-base-coeff",
        type=float,
        default=0.0,
        help="Explicit KL penalty to the initial trainable adapter snapshot; v2 starts at 0.05.",
    )
    parser.add_argument("--kl-to-base-temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-early-stop-window",
        type=int,
        default=0,
        help="Stop after an update when the mean unscaled KL over this many recent steps exceeds the threshold.",
    )
    parser.add_argument(
        "--kl-early-stop-mean-threshold",
        type=float,
        default=0.0,
        help="Mean unscaled KL threshold for --kl-early-stop-window; disabled at <=0.",
    )
    parser.add_argument("--replay-max-prompt-tokens", type=int, default=768)
    parser.add_argument("--replay-max-seq-tokens", type=int, default=1024)
    parser.add_argument("--retention-probe-every-steps", type=int, default=50)
    parser.add_argument("--retention-probe-limit", type=int, default=5)
    parser.add_argument("--retention-collapse-flex-accuracy", type=float, default=0.40)
    parser.add_argument("--retention-gsm8k-path", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--retention-gsm8k-fewshot-path", type=Path, default=DEFAULT_GSM8K_FEWSHOT)
    parser.add_argument("--retention-gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--retention-block-size", type=int, default=32)
    parser.add_argument("--retention-small-block-size", type=int, default=32)
    parser.add_argument("--retention-max-new-tokens", type=int, default=384)
    parser.add_argument("--retention-threshold", type=float, default=0.9)
    parser.add_argument("--retention-top-p", type=float, default=0.95)
    parser.add_argument("--retention-temperature", type=float, default=0.0)
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
