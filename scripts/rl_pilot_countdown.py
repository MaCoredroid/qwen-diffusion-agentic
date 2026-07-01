#!/usr/bin/env python3
"""Minimal Countdown diffu-GRPO de-risk pilot.

This is intentionally a feasibility script, not a polished trainer. It uses
reasoning-gym Countdown rows, generates group rollouts with the local
block-diffusion model plus a live arithmetic grammar, scores strict
reasoning-gym success, and applies a GRPO-style LoRA policy-gradient update
over the constrained token distribution.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models/qwen3.5-9b-fastdllm-init"
DEFAULT_ADAPTER = ROOT / "runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000"
DEFAULT_OUT = ROOT / "runs/rl_pilot_countdown"

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from flare_hf_cache import RequestDiffusionState


def configure_cuda_env() -> None:
    venv_root = Path(sys.executable).resolve().parents[1]
    cuda_root = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    if not cuda_root.exists():
        cuda_root = ROOT / ".venv-fastdllm/lib/python3.10/site-packages/nvidia/cu13"
    if cuda_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))
        os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ["FASTDLLM_GDN_KERNEL"] = "torch"
    os.environ["FASTDLLM_FLARE_GDN_ROUTE"] = "route_i"
    os.environ["FASTDLLM_FLARE_TWO_STREAM"] = "1"
    os.environ["FLARE_TWO_STREAM"] = "1"


class GpuMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if proc.returncode == 0 and proc.stdout.strip():
                    parts = [part.strip() for part in proc.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        self.samples.append(
                            {
                                "timestamp": parts[0],
                                "memory_mib": int(float(parts[1])),
                                "util_pct": int(float(parts[2])),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "samples": 0,
                "gpu_peak_memory_mib": None,
                "gpu_util_mean_pct": None,
                "gpu_util_max_pct": None,
            }
        utils = [sample["util_pct"] for sample in self.samples]
        mem = [sample["memory_mib"] for sample in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_peak_memory_mib": max(mem),
            "gpu_util_mean_pct": sum(utils) / len(utils),
            "gpu_util_max_pct": max(utils),
        }


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def apply_chat_template(tokenizer, messages: list[dict[str, str]], **kwargs):
    kwargs = dict(kwargs)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_countdown_prompt(tokenizer, entry: dict[str, Any]) -> str:
    messages = [{"role": "user", "content": entry["question"]}]
    return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)


@dataclass
class TokenGrammar:
    char_to_id: dict[str, int]
    id_to_char: dict[int, str]
    stop_ids: list[int]
    mask_id: int


def build_token_grammar(tokenizer, mask_id: int, stop_ids: list[int]) -> TokenGrammar:
    char_to_id: dict[str, int] = {}
    for ch in "0123456789+-*/()":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Countdown grammar needs single-token literal {ch!r}, got {ids}")
        char_to_id[ch] = int(ids[0])
    stop_ids = [int(item) for item in stop_ids if item is not None]
    stop_ids = list(dict.fromkeys(stop_ids))
    return TokenGrammar(
        char_to_id=char_to_id,
        id_to_char={token_id: ch for ch, token_id in char_to_id.items()},
        stop_ids=stop_ids,
        mask_id=int(mask_id),
    )


@dataclass
class CountdownGrammarState:
    numbers: list[int]
    remaining: Counter[str] = field(init=False)
    open_parens: int = 0
    expect_value: bool = True
    pending_number: str = ""
    stopped: bool = False
    token_ids: list[int] = field(default_factory=list)
    expression_chars: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining = Counter(str(int(number)) for number in self.numbers)

    @property
    def expression(self) -> str:
        return "".join(self.expression_chars)

    @property
    def emitted_count(self) -> int:
        return len(self.token_ids)

    def remaining_total(self) -> int:
        return sum(self.remaining.values())

    def _prefix_exists(self, prefix: str) -> bool:
        return any(count > 0 and number.startswith(prefix) for number, count in self.remaining.items())

    def _can_commit_pending(self) -> bool:
        return bool(self.pending_number) and self.remaining.get(self.pending_number, 0) > 0

    def allowed_token_ids(self, grammar: TokenGrammar) -> list[int]:
        if self.stopped:
            return []

        allowed_chars: set[str] = set()
        allow_stop = False
        remaining = self.remaining_total()

        if self.pending_number:
            for digit in "0123456789":
                if self._prefix_exists(self.pending_number + digit):
                    allowed_chars.add(digit)
            if self._can_commit_pending():
                remaining_after = remaining - 1
                if remaining_after > 0:
                    allowed_chars.update("+-*/")
                if self.open_parens > 0:
                    allowed_chars.add(")")
                if remaining_after == 0 and self.open_parens == 0:
                    allow_stop = True
        elif self.expect_value:
            if remaining > 0:
                if self.open_parens < max(1, len(self.numbers)):
                    allowed_chars.add("(")
                for digit in "0123456789":
                    if self._prefix_exists(digit):
                        allowed_chars.add(digit)
        else:
            if remaining > 0:
                allowed_chars.update("+-*/")
            if self.open_parens > 0:
                allowed_chars.add(")")
            if remaining == 0 and self.open_parens == 0:
                allow_stop = True

        ids = [grammar.char_to_id[ch] for ch in sorted(allowed_chars)]
        if allow_stop:
            ids.extend(grammar.stop_ids)
        return list(dict.fromkeys(ids))

    def _commit_pending_number(self) -> None:
        if not self._can_commit_pending():
            raise ValueError(f"Cannot commit pending number {self.pending_number!r}")
        self.remaining[self.pending_number] -= 1
        if self.remaining[self.pending_number] <= 0:
            del self.remaining[self.pending_number]
        self.pending_number = ""
        self.expect_value = False

    def advance(self, token_id: int, grammar: TokenGrammar) -> None:
        token_id = int(token_id)
        self.token_ids.append(token_id)
        if token_id in grammar.stop_ids:
            if self.pending_number:
                self._commit_pending_number()
            if self.remaining_total() != 0 or self.open_parens != 0 or self.expect_value:
                raise ValueError("Countdown grammar stopped from an incomplete state")
            self.stopped = True
            return

        ch = grammar.id_to_char.get(token_id)
        if ch is None:
            raise ValueError(f"Token {token_id} is not in the Countdown grammar")
        self.expression_chars.append(ch)

        if self.pending_number:
            if ch.isdigit():
                self.pending_number += ch
                return
            self._commit_pending_number()
            if ch in "+-*/":
                self.expect_value = True
                return
            if ch == ")":
                self.open_parens -= 1
                if self.open_parens < 0:
                    raise ValueError("Countdown grammar closed too many parentheses")
                self.expect_value = False
                return
            raise ValueError(f"Unexpected token after pending number: {ch!r}")

        if self.expect_value:
            if ch == "(":
                self.open_parens += 1
                self.expect_value = True
                return
            if ch.isdigit():
                self.pending_number = ch
                self.expect_value = True
                return
            raise ValueError(f"Expected value, got {ch!r}")

        if ch in "+-*/":
            self.expect_value = True
            return
        if ch == ")":
            self.open_parens -= 1
            if self.open_parens < 0:
                raise ValueError("Countdown grammar closed too many parentheses")
            self.expect_value = False
            return
        raise ValueError(f"Expected operator/close, got {ch!r}")


@dataclass
class RolloutStep:
    input_ids: torch.Tensor
    row_indices: list[int]
    positions: list[int]
    selected_token_ids: list[int]
    allowed_token_ids: list[list[int]]


@dataclass
class RolloutResult:
    prompt_idx: int
    prompt_entry: dict[str, Any]
    expressions: list[str]
    token_ids: list[list[int]]
    rg_scores: list[float]
    strict_rewards: list[float]
    rewards: list[float]
    steps: list[RolloutStep]
    seconds: float
    denoise_forwards: int
    cache_read_calls: int = 0
    cache_advance_calls: int = 0


@dataclass
class RawRolloutResult:
    prompt_idx: int
    prompt_entry: dict[str, Any]
    texts: list[str]
    token_ids: list[list[int]]
    rg_scores: list[float]
    strict_rewards: list[float]
    rewards: list[float]
    steps: list[RolloutStep]
    seconds: float
    denoise_forwards: int
    cache_read_calls: int = 0
    cache_advance_calls: int = 0


def masked_allowed_logprob(
    row_logits: torch.Tensor,
    selected_token_id: int,
    allowed_token_ids: list[int],
    temperature: float,
) -> torch.Tensor:
    allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=row_logits.device)
    logits = row_logits.float()
    if temperature > 0:
        logits = logits / float(temperature)
    selected = torch.tensor(int(selected_token_id), dtype=torch.long, device=row_logits.device)
    denom = torch.logsumexp(logits.index_select(0, allowed), dim=0)
    return logits.index_select(0, selected.view(1))[0] - denom


def raw_full_vocab_logprob(
    row_logits: torch.Tensor,
    selected_token_id: int,
    *,
    temperature: float,
    mask_id: int,
) -> torch.Tensor:
    logits = row_logits.float().clone()
    logits[int(mask_id)] = -torch.inf
    if temperature > 0:
        logits = logits / float(temperature)
    selected = torch.tensor(int(selected_token_id), dtype=torch.long, device=row_logits.device)
    return logits.index_select(0, selected.view(1))[0] - torch.logsumexp(logits, dim=0)


def sample_from_allowed(
    row_logits: torch.Tensor,
    allowed_token_ids: list[int],
    temperature: float,
    generator: torch.Generator,
) -> int:
    allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=row_logits.device)
    logits = row_logits.float().index_select(0, allowed)
    if temperature <= 0:
        return int(allowed[int(torch.argmax(logits).item())].item())
    probs = torch.softmax(logits / float(temperature), dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(allowed[int(sampled.item())].item())


def shift_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def strict_countdown_score(dataset, expression: str, entry: dict[str, Any]) -> tuple[float, float]:
    rg_score = float(dataset.score_answer(expression, entry))
    strict = 1.0 if rg_score >= 1.0 - 1e-9 else 0.0
    return rg_score, strict


def graded_countdown_reward(dataset, expression: str, entry: dict[str, Any]) -> tuple[float, float, float]:
    """Return reasoning-gym score, strict success, and dense Countdown reward.

    reasoning-gym gives useful partial credit for parseable all-number answers,
    but most wrong all-number expressions tie at 0.05.  This Stage-1 reward
    keeps exact success at 1.0 and adds bounded inverse-distance credit for
    parseable expressions that use exactly the required number multiset.
    """
    rg_score, strict = strict_countdown_score(dataset, expression, entry)
    reward = float(rg_score)
    if strict:
        return rg_score, strict, 1.0

    if expression is None or not str(expression).strip():
        return rg_score, strict, reward

    try:
        from reasoning_gym.games.countdown import _extract_ints, parse_expr

        value = float(parse_expr(expression))
        used_numbers = _extract_ints(expression)
        target_numbers = entry["metadata"]["numbers"]
        if sorted(used_numbers) != sorted(target_numbers):
            return rg_score, strict, max(reward, 0.02)
        distance = abs(value - float(entry["metadata"]["target"]))
        shaped = 0.05 + 0.45 / (1.0 + distance)
        reward = max(reward, min(0.5, shaped))
    except Exception:
        pass
    return rg_score, strict, float(reward)


def constrained_countdown_rollout(
    model,
    tokenizer,
    dataset,
    entry: dict[str, Any],
    prompt_idx: int,
    grammar: TokenGrammar,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    record_steps: bool,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
) -> RolloutResult:
    prompt = make_countdown_prompt(tokenizer, entry)
    prompt_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    original_len = int(prompt_ids.shape[1])
    input_ids = prompt_ids.repeat(group_size, 1)
    output_ids = input_ids
    states = [CountdownGrammarState(list(entry["metadata"]["numbers"])) for _ in range(group_size)]
    steps: list[RolloutStep] = []
    denoise_forwards = 0
    cache_state = RequestDiffusionState.reset(model, output_ids, block_size) if use_fast_cache else None
    stop_fill_id = int(grammar.stop_ids[0]) if grammar.stop_ids else int(grammar.mask_id)

    sync_cuda()
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        while output_ids.shape[1] - original_len < max_new_tokens:
            active_rows = [row for row, state in enumerate(states) if not state.stopped]
            if not active_rows:
                break
            if use_fast_cache:
                active_len = output_ids.shape[1] - cache_state.block_start
                if active_len < 0 or active_len >= block_size:
                    raise RuntimeError(
                        f"invalid Countdown cache active_len={active_len} "
                        f"output_len={output_ids.shape[1]} block_start={cache_state.block_start}"
                    )
                block_pad = int(block_size) - int(active_len)
            else:
                block_pad = int(block_size) - (output_ids.shape[1] % int(block_size))
                if block_pad == 0:
                    block_pad = int(block_size)
            remaining = int(max_new_tokens) - (output_ids.shape[1] - original_len)
            block_pad = min(block_pad, remaining)
            if block_pad <= 0:
                break
            masks = torch.full(
                (group_size, block_pad),
                grammar.mask_id,
                dtype=torch.long,
                device=output_ids.device,
            )
            x_t = torch.cat([output_ids, masks], dim=1)

            while bool((x_t[:, -block_pad:] == grammar.mask_id).any().item()):
                active_rows = [row for row, state in enumerate(states) if not state.stopped]
                if not active_rows:
                    break
                if use_fast_cache:
                    logits = cache_state.shifted_active_logits(model, x_t)
                    logit_offset = int(cache_state.block_start)
                else:
                    output = model(input_ids=x_t, use_cache=False)
                    logits = shift_logits(output.logits)
                    logit_offset = 0
                denoise_forwards += 1
                before = x_t.detach().cpu() if record_steps else None

                row_indices: list[int] = []
                positions: list[int] = []
                selected_token_ids: list[int] = []
                allowed_lists: list[list[int]] = []
                for row in active_rows:
                    state = states[row]
                    pos = original_len + state.emitted_count
                    if pos >= x_t.shape[1]:
                        continue
                    allowed = state.allowed_token_ids(grammar)
                    if not allowed:
                        continue
                    token_id = sample_from_allowed(logits[row, pos - logit_offset], allowed, temperature, generator)
                    row_indices.append(row)
                    positions.append(pos)
                    selected_token_ids.append(token_id)
                    allowed_lists.append(allowed)

                if record_steps and before is not None and positions:
                    steps.append(
                        RolloutStep(
                            input_ids=before,
                            row_indices=row_indices,
                            positions=positions,
                            selected_token_ids=selected_token_ids,
                            allowed_token_ids=allowed_lists,
                        )
                    )

                if not positions:
                    break

                for row, pos, token_id in zip(row_indices, positions, selected_token_ids):
                    try:
                        states[row].advance(token_id, grammar)
                    except ValueError:
                        states[row].stopped = True
                    x_t[row, pos] = int(token_id)
                    if states[row].stopped and pos + 1 < x_t.shape[1]:
                        x_t[row, pos + 1 :] = stop_fill_id

            if use_fast_cache:
                active_block = x_t[:, cache_state.block_start :]
                if active_block.shape[1] == block_size and not bool((active_block == grammar.mask_id).any().item()):
                    cache_state.advance(model, active_block)
            output_ids = x_t

    sync_cuda()
    seconds = time.perf_counter() - started

    expressions = [state.expression for state in states]
    token_ids = [state.token_ids for state in states]
    rg_scores: list[float] = []
    strict_rewards: list[float] = []
    rewards: list[float] = []
    for expression in expressions:
        rg_score, strict, reward = graded_countdown_reward(dataset, expression, entry)
        rg_scores.append(rg_score)
        strict_rewards.append(strict)
        rewards.append(reward)

    cache_read_calls = 0 if cache_state is None else int(cache_state.read_calls)
    cache_advance_calls = 0 if cache_state is None else int(cache_state.advance_calls)
    return RolloutResult(
        prompt_idx=prompt_idx,
        prompt_entry=entry,
        expressions=expressions,
        token_ids=token_ids,
        rg_scores=rg_scores,
        strict_rewards=strict_rewards,
        rewards=rewards,
        steps=steps,
        seconds=seconds,
        denoise_forwards=denoise_forwards,
        cache_read_calls=cache_read_calls,
        cache_advance_calls=cache_advance_calls,
    )


def raw_diffusion_generate_one(
    model,
    tokenizer,
    entry: dict[str, Any],
    grammar: TokenGrammar,
    *,
    max_new_tokens: int,
    threshold: float,
    temperature: float,
    top_p: float,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
) -> tuple[str, dict[str, Any]]:
    prompt = make_countdown_prompt(tokenizer, entry)
    prompt_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    original_len = int(prompt_ids.shape[1])
    output_ids = prompt_ids
    stop_ids = torch.tensor(grammar.stop_ids, dtype=torch.long, device=prompt_ids.device)
    denoise_forwards = 0
    selected_mask_tokens = 0
    cache_state = RequestDiffusionState.reset(model, output_ids, block_size) if use_fast_cache else None

    def stopped_prefix(sequence: torch.Tensor) -> torch.Tensor | None:
        generated = sequence[:, original_len:]
        stop_mask = torch.isin(generated, stop_ids)
        if not bool(stop_mask.any().item()):
            return None
        first_stop = int(stop_mask.nonzero(as_tuple=False)[0, 1].item())
        if bool((generated[:, :first_stop] == grammar.mask_id).any().item()):
            return None
        return generated[:, :first_stop]

    model.eval()
    with torch.no_grad():
        while output_ids.shape[1] - original_len < max_new_tokens:
            stopped = stopped_prefix(output_ids)
            if stopped is not None:
                new_ids = stopped[0]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                return text, {
                    "denoise_forwards": denoise_forwards,
                    "selected_mask_tokens": selected_mask_tokens,
                    "generated_tokens": int(new_ids.numel()),
                    "stopped": True,
                    "cache_read_calls": 0 if cache_state is None else int(cache_state.read_calls),
                    "cache_advance_calls": 0 if cache_state is None else int(cache_state.advance_calls),
                }
            if use_fast_cache:
                active_len = output_ids.shape[1] - cache_state.block_start
                if active_len < 0 or active_len >= block_size:
                    raise RuntimeError(
                        f"invalid raw cache active_len={active_len} "
                        f"output_len={output_ids.shape[1]} block_start={cache_state.block_start}"
                    )
                block_pad = int(block_size) - int(active_len)
            else:
                block_pad = int(block_size) - (output_ids.shape[1] % int(block_size))
                if block_pad == 0:
                    block_pad = int(block_size)
            remaining = int(max_new_tokens) - (output_ids.shape[1] - original_len)
            block_pad = min(block_pad, remaining)
            if block_pad <= 0:
                break
            masks = torch.full((1, block_pad), grammar.mask_id, dtype=torch.long, device=prompt_ids.device)
            x_t = torch.cat([output_ids, masks], dim=1)

            while bool((x_t[:, -block_pad:] == grammar.mask_id).any().item()):
                if use_fast_cache:
                    logits = cache_state.shifted_active_logits(model, x_t)
                    logit_offset = int(cache_state.block_start)
                    current_mask = x_t[:, cache_state.block_start :] == grammar.mask_id
                else:
                    output = model(input_ids=x_t, use_cache=False)
                    logits = shift_logits(output.logits)
                    logit_offset = 0
                    current_mask = x_t == grammar.mask_id
                logits = logits.clone()
                logits[..., grammar.mask_id] = torch.finfo(logits.dtype).min
                if temperature <= 0:
                    probs = torch.softmax(logits.float(), dim=-1)
                    x_1 = probs.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        remove = cumulative > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        full_remove = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, sorted_indices, remove)
                        probs = probs.masked_fill(full_remove, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    x_1 = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]),
                        1,
                        generator=generator,
                    ).view(probs.shape[:-1])
                x1_p = torch.squeeze(torch.gather(probs, dim=-1, index=x_1.unsqueeze(-1)), -1)
                active_probs = torch.where(current_mask, x1_p, torch.full_like(x1_p, -torch.inf))
                if not bool(torch.isfinite(active_probs).any().item()):
                    break
                unmask_idx = active_probs > threshold
                max_prob_idx = active_probs.argmax(dim=-1)
                unmask_idx[torch.arange(x_1.shape[0], device=x_1.device), max_prob_idx] = True
                unmask_idx = unmask_idx & current_mask
                selected_mask_tokens += int(((x_1 == grammar.mask_id) & unmask_idx).sum().item())
                span = x_t[:, logit_offset:].clone()
                span[unmask_idx] = x_1[unmask_idx]
                x_t[:, logit_offset:] = span
                denoise_forwards += 1
                stopped = stopped_prefix(x_t)
                if stopped is not None:
                    output_ids = x_t
                    break

            if use_fast_cache:
                active_block = x_t[:, cache_state.block_start :]
                if active_block.shape[1] == block_size and not bool((active_block == grammar.mask_id).any().item()):
                    cache_state.advance(model, active_block)
            output_ids = x_t
            if stopped_prefix(output_ids) is not None:
                break

    stopped = stopped_prefix(output_ids)
    new_ids = stopped[0] if stopped is not None else output_ids[0, original_len:]
    new_ids = new_ids[new_ids != grammar.mask_id]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, {
        "denoise_forwards": denoise_forwards,
        "selected_mask_tokens": selected_mask_tokens,
        "generated_tokens": int(new_ids.numel()),
        "stopped": stopped is not None,
        "cache_read_calls": 0 if cache_state is None else int(cache_state.read_calls),
        "cache_advance_calls": 0 if cache_state is None else int(cache_state.advance_calls),
    }


def raw_diffusion_rollout(
    model,
    tokenizer,
    dataset,
    entry: dict[str, Any],
    prompt_idx: int,
    grammar: TokenGrammar,
    *,
    group_size: int,
    max_new_tokens: int,
    threshold: float,
    temperature: float,
    top_p: float,
    record_steps: bool,
    use_fast_cache: bool,
    block_size: int,
    generator: torch.Generator,
) -> RawRolloutResult:
    prompt = make_countdown_prompt(tokenizer, entry)
    prompt_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    original_len = int(prompt_ids.shape[1])
    output_ids = prompt_ids.repeat(group_size, 1)
    stop_ids = torch.tensor(grammar.stop_ids, dtype=torch.long, device=prompt_ids.device)
    stop_fill_id = int(grammar.stop_ids[0]) if grammar.stop_ids else int(grammar.mask_id)
    stopped_rows = torch.zeros(group_size, dtype=torch.bool, device=prompt_ids.device)
    steps: list[RolloutStep] = []
    denoise_forwards = 0
    cache_state = RequestDiffusionState.reset(model, output_ids, block_size) if use_fast_cache else None

    def complete_stop_positions(sequence: torch.Tensor) -> list[int | None]:
        generated = sequence[:, original_len:]
        stop_mask = torch.isin(generated, stop_ids)
        out: list[int | None] = []
        for row in range(sequence.shape[0]):
            row_stops = stop_mask[row].nonzero(as_tuple=False)
            found = None
            for item in row_stops:
                rel = int(item.item())
                if not bool((generated[row, :rel] == grammar.mask_id).any().item()):
                    found = rel
                    break
            out.append(found)
        return out

    def mark_stopped(sequence: torch.Tensor) -> torch.Tensor:
        stop_positions = complete_stop_positions(sequence)
        for row, rel in enumerate(stop_positions):
            if rel is None:
                continue
            stopped_rows[row] = True
            abs_after = original_len + rel + 1
            if abs_after < sequence.shape[1]:
                sequence[row, abs_after:] = stop_fill_id
        return sequence

    sync_cuda()
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        while output_ids.shape[1] - original_len < max_new_tokens:
            output_ids = mark_stopped(output_ids)
            if bool(stopped_rows.all().item()):
                break
            if use_fast_cache:
                active_len = output_ids.shape[1] - cache_state.block_start
                if active_len < 0 or active_len >= block_size:
                    raise RuntimeError(
                        f"invalid raw rollout cache active_len={active_len} "
                        f"output_len={output_ids.shape[1]} block_start={cache_state.block_start}"
                    )
                block_pad = int(block_size) - int(active_len)
            else:
                block_pad = int(block_size) - (output_ids.shape[1] % int(block_size))
                if block_pad == 0:
                    block_pad = int(block_size)
            remaining = int(max_new_tokens) - (output_ids.shape[1] - original_len)
            block_pad = min(block_pad, remaining)
            if block_pad <= 0:
                break
            masks = torch.full(
                (group_size, block_pad),
                grammar.mask_id,
                dtype=torch.long,
                device=output_ids.device,
            )
            masks[stopped_rows] = stop_fill_id
            x_t = torch.cat([output_ids, masks], dim=1)

            while bool(((x_t[:, -block_pad:] == grammar.mask_id) & ~stopped_rows[:, None]).any().item()):
                if use_fast_cache:
                    logits = cache_state.shifted_active_logits(model, x_t)
                    logit_offset = int(cache_state.block_start)
                    current_mask = x_t[:, cache_state.block_start :] == grammar.mask_id
                else:
                    output = model(input_ids=x_t, use_cache=False)
                    logits = shift_logits(output.logits)
                    logit_offset = 0
                    current_mask = x_t == grammar.mask_id
                current_mask = current_mask & ~stopped_rows[:, None]
                logits = logits.clone()
                logits[..., grammar.mask_id] = torch.finfo(logits.dtype).min
                if temperature <= 0:
                    probs = torch.softmax(logits.float(), dim=-1)
                    x_1 = probs.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits.float() / float(temperature), dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        remove = cumulative > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        full_remove = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, sorted_indices, remove)
                        probs = probs.masked_fill(full_remove, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    x_1 = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]),
                        1,
                        generator=generator,
                    ).view(probs.shape[:-1])
                x1_p = torch.squeeze(torch.gather(probs, dim=-1, index=x_1.unsqueeze(-1)), -1)
                active_probs = torch.where(current_mask, x1_p, torch.full_like(x1_p, -torch.inf))
                if not bool(torch.isfinite(active_probs).any().item()):
                    break
                unmask_idx = active_probs > threshold
                for row in range(group_size):
                    if bool(stopped_rows[row].item()) or not bool(current_mask[row].any().item()):
                        continue
                    max_prob_idx = int(active_probs[row].argmax().item())
                    unmask_idx[row, max_prob_idx] = True
                unmask_idx = unmask_idx & current_mask

                before = x_t.detach().cpu() if record_steps else None
                if record_steps and before is not None and bool(unmask_idx.any().item()):
                    row_indices: list[int] = []
                    positions: list[int] = []
                    selected_token_ids: list[int] = []
                    for row, local_pos in unmask_idx.nonzero(as_tuple=False).tolist():
                        row_indices.append(int(row))
                        positions.append(int(logit_offset + local_pos))
                        selected_token_ids.append(int(x_1[row, local_pos].item()))
                    steps.append(
                        RolloutStep(
                            input_ids=before,
                            row_indices=row_indices,
                            positions=positions,
                            selected_token_ids=selected_token_ids,
                            allowed_token_ids=[[] for _ in selected_token_ids],
                        )
                    )

                span = x_t[:, logit_offset:].clone()
                span[unmask_idx] = x_1[unmask_idx]
                x_t[:, logit_offset:] = span
                denoise_forwards += 1
                x_t = mark_stopped(x_t)

            if use_fast_cache:
                active_block = x_t[:, cache_state.block_start :]
                if active_block.shape[1] == block_size and not bool((active_block == grammar.mask_id).any().item()):
                    cache_state.advance(model, active_block)
            output_ids = x_t

    sync_cuda()
    seconds = time.perf_counter() - started
    stop_positions = complete_stop_positions(output_ids)
    texts: list[str] = []
    token_ids: list[list[int]] = []
    rg_scores: list[float] = []
    strict_rewards: list[float] = []
    rewards: list[float] = []
    for row, rel_stop in enumerate(stop_positions):
        generated = output_ids[row, original_len:]
        if rel_stop is not None:
            generated = generated[:rel_stop]
        generated = generated[generated != grammar.mask_id]
        row_token_ids = [int(item) for item in generated.detach().cpu().tolist()]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        rg_score, strict, reward = graded_countdown_reward(dataset, text, entry)
        texts.append(text)
        token_ids.append(row_token_ids)
        rg_scores.append(rg_score)
        strict_rewards.append(strict)
        rewards.append(reward)

    cache_read_calls = 0 if cache_state is None else int(cache_state.read_calls)
    cache_advance_calls = 0 if cache_state is None else int(cache_state.advance_calls)
    return RawRolloutResult(
        prompt_idx=prompt_idx,
        prompt_entry=entry,
        texts=texts,
        token_ids=token_ids,
        rg_scores=rg_scores,
        strict_rewards=strict_rewards,
        rewards=rewards,
        steps=steps,
        seconds=seconds,
        denoise_forwards=denoise_forwards,
        cache_read_calls=cache_read_calls,
        cache_advance_calls=cache_advance_calls,
    )


def grpo_advantages(rewards: list[float]) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device="cuda")
    mean = values.mean()
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return (values - mean) / std.clamp_min(1e-6)


def rollout_policy_token_count(rollout: RolloutResult) -> int:
    return sum(len(step.positions) for step in rollout.steps)


def raw_rollout_policy_token_count(rollout: RawRolloutResult) -> int:
    return sum(len(step.positions) for step in rollout.steps)


def strict_correct_list(rollout: RolloutResult) -> list[bool]:
    return [float(item) >= 1.0 - 1e-9 for item in rollout.strict_rewards]


def backward_dual_term_loss(
    model,
    rollout: RolloutResult,
    advantages: torch.Tensor,
    temperature: float,
    *,
    lambda_raw: float,
    rescore_micro_batch_size: int,
) -> dict[str, Any]:
    """Backprop constrained GRPO plus raw CE self-distillation.

    The rollout was sampled by the cached serving path, but every differentiated
    logprob is replayed through the training forward.  The policy term masks the
    distribution to the live grammar.  The raw-internalization term uses the
    same verified-correct decoder trajectory tokens with the full vocabulary.
    """
    token_count = rollout_policy_token_count(rollout)
    correct = strict_correct_list(rollout)
    raw_token_count = 0
    for step in rollout.steps:
        raw_token_count += sum(1 for row in step.row_indices if correct[row])
    if token_count <= 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "raw_internalize_loss": 0.0,
            "policy_tokens": 0,
            "raw_internalize_tokens": 0,
            "raw_internalize_samples": sum(int(item) for item in correct),
            "mean_logprob": None,
            "mean_raw_logprob": None,
        }

    per_sample_tokens = torch.zeros_like(advantages)
    logp_sum = torch.zeros_like(advantages)
    raw_logp_sum = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    policy_loss_sum_t = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    raw_loss_sum_t = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    micro_batch_size = max(1, int(rescore_micro_batch_size))
    pending: list[tuple[torch.Tensor, int, int, int, list[int]]] = []

    def flush_pending() -> None:
        nonlocal raw_logp_sum, policy_loss_sum_t, raw_loss_sum_t, pending
        if not pending:
            return
        x = torch.cat([item[0] for item in pending], dim=0).to("cuda", non_blocking=True)
        rows = [item[1] for item in pending]
        positions = [item[2] for item in pending]
        selected = [item[3] for item in pending]
        allowed_lists = [item[4] for item in pending]
        output = model(input_ids=x, use_cache=False)
        logits = shift_logits(output.logits)
        terms = []
        for local_idx, (row, pos, token_id, allowed) in enumerate(
            zip(rows, positions, selected, allowed_lists)
        ):
            row_logits = logits[local_idx, pos]
            logp = masked_allowed_logprob(row_logits, token_id, allowed, temperature)
            policy_unscaled = -advantages[row] * logp
            terms.append(policy_unscaled / float(token_count))
            policy_loss_sum_t = policy_loss_sum_t + policy_unscaled.detach().float()
            per_sample_tokens[row] += 1
            logp_sum[row] += logp.detach()

            if lambda_raw > 0 and raw_token_count > 0 and correct[row]:
                raw_logp = torch.log_softmax(row_logits.float(), dim=-1)[int(token_id)]
                raw_unscaled = -raw_logp
                terms.append(float(lambda_raw) * raw_unscaled / float(raw_token_count))
                raw_loss_sum_t = raw_loss_sum_t + raw_unscaled.detach().float()
                raw_logp_sum = raw_logp_sum + raw_logp.detach()
        if terms:
            sum(terms).backward()
        del output, logits, x, terms
        pending = []

    for step in rollout.steps:
        items = list(
            zip(
                step.row_indices,
                step.positions,
                step.selected_token_ids,
                step.allowed_token_ids,
            )
        )
        for row, pos, token_id, allowed in items:
            row_input = step.input_ids[row : row + 1]
            if pending and (
                pending[0][0].shape[1] != row_input.shape[1] or len(pending) >= micro_batch_size
            ):
                flush_pending()
            pending.append((row_input, row, pos, token_id, allowed))
    flush_pending()
    mean_logprob = float((logp_sum.sum() / per_sample_tokens.sum().clamp_min(1)).item())
    mean_raw_logprob = (
        float((raw_logp_sum / max(1, raw_token_count)).item())
        if raw_token_count > 0
        else None
    )
    policy_loss_sum = float(policy_loss_sum_t.detach().cpu().item())
    raw_loss_sum = float(raw_loss_sum_t.detach().cpu().item())
    return {
        "loss": (policy_loss_sum / float(token_count)) + float(lambda_raw) * (raw_loss_sum / max(1, raw_token_count)),
        "policy_loss": policy_loss_sum / float(token_count),
        "raw_internalize_loss": raw_loss_sum / max(1, raw_token_count) if raw_token_count > 0 else 0.0,
        "policy_tokens": token_count,
        "raw_internalize_tokens": raw_token_count,
        "raw_internalize_samples": sum(int(item) for item in correct),
        "mean_logprob": mean_logprob,
        "mean_raw_logprob": mean_raw_logprob,
    }


def backward_raw_rollout_policy_loss(
    model,
    rollout: RawRolloutResult,
    advantages: torch.Tensor,
    *,
    temperature: float,
    mask_id: int,
    raw_rl_weight: float,
    rescore_micro_batch_size: int,
) -> dict[str, Any]:
    """Backprop raw no-decoder GRPO tokens through the training forward."""
    token_count = raw_rollout_policy_token_count(rollout)
    if token_count <= 0 or raw_rl_weight <= 0:
        return {
            "raw_rl_loss": 0.0,
            "raw_rl_tokens": token_count,
            "raw_rl_mean_logprob": None,
        }

    per_sample_tokens = torch.zeros_like(advantages)
    logp_sum = torch.zeros_like(advantages)
    loss_sum_t = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    micro_batch_size = max(1, int(rescore_micro_batch_size))
    pending: list[tuple[torch.Tensor, int, int, int]] = []

    def flush_pending() -> None:
        nonlocal loss_sum_t, pending
        if not pending:
            return
        x = torch.cat([item[0] for item in pending], dim=0).to("cuda", non_blocking=True)
        rows = [item[1] for item in pending]
        positions = [item[2] for item in pending]
        selected = [item[3] for item in pending]
        output = model(input_ids=x, use_cache=False)
        logits = shift_logits(output.logits)
        terms = []
        for local_idx, (row, pos, token_id) in enumerate(zip(rows, positions, selected)):
            logp = raw_full_vocab_logprob(
                logits[local_idx, pos],
                token_id,
                temperature=temperature,
                mask_id=mask_id,
            )
            unscaled = -advantages[row] * logp
            terms.append(float(raw_rl_weight) * unscaled / float(token_count))
            loss_sum_t = loss_sum_t + unscaled.detach().float()
            per_sample_tokens[row] += 1
            logp_sum[row] += logp.detach()
        if terms:
            sum(terms).backward()
        del output, logits, x, terms
        pending = []

    for step in rollout.steps:
        for row, pos, token_id in zip(step.row_indices, step.positions, step.selected_token_ids):
            row_input = step.input_ids[row : row + 1]
            if pending and (
                pending[0][0].shape[1] != row_input.shape[1] or len(pending) >= micro_batch_size
            ):
                flush_pending()
            pending.append((row_input, row, pos, token_id))
    flush_pending()

    mean_logprob = float((logp_sum.sum() / per_sample_tokens.sum().clamp_min(1)).item())
    raw_rl_loss = float(loss_sum_t.detach().cpu().item()) / float(token_count)
    return {
        "raw_rl_loss": raw_rl_loss,
        "raw_rl_tokens": token_count,
        "raw_rl_mean_logprob": mean_logprob,
    }


def evaluate_lanes(
    model,
    tokenizer,
    dataset,
    entries: list[dict[str, Any]],
    grammar: TokenGrammar,
    args,
    generator: torch.Generator,
) -> dict[str, Any]:
    raw_rows = []
    constrained_rows = []
    raw_correct = 0
    constrained_correct = 0
    raw_rg_sum = 0.0
    constrained_rg_sum = 0.0
    raw_graded_sum = 0.0
    constrained_graded_sum = 0.0
    started = time.perf_counter()
    model.eval()
    for idx, entry in enumerate(entries):
        raw_text, raw_metrics = raw_diffusion_generate_one(
            model,
            tokenizer,
            entry,
            grammar,
            max_new_tokens=args.eval_max_new_tokens,
            threshold=args.raw_threshold,
            temperature=args.raw_temperature,
            top_p=args.raw_top_p,
            use_fast_cache=args.use_fast_serving_cache,
            block_size=args.block_size,
            generator=generator,
        )
        raw_rg, raw_strict, raw_reward = graded_countdown_reward(dataset, raw_text, entry)
        raw_correct += int(raw_strict)
        raw_rg_sum += raw_rg
        raw_graded_sum += raw_reward
        raw_rows.append(
            {
                "idx": idx,
                "target": entry["metadata"]["target"],
                "numbers": entry["metadata"]["numbers"],
                "gold": entry["answer"],
                "raw": raw_text,
                "rg_score": raw_rg,
                "strict": raw_strict,
                "graded_reward": raw_reward,
                "sampler": raw_metrics,
            }
        )

        constrained = constrained_countdown_rollout(
            model,
            tokenizer,
            dataset,
            entry,
            idx,
            grammar,
            group_size=1,
            max_new_tokens=args.eval_max_new_tokens,
            temperature=args.eval_constrained_temperature,
            record_steps=False,
            use_fast_cache=args.use_fast_serving_cache,
            block_size=args.block_size,
            generator=generator,
        )
        con_text = constrained.expressions[0]
        con_rg = constrained.rg_scores[0]
        con_strict = constrained.strict_rewards[0]
        con_reward = constrained.rewards[0]
        constrained_correct += int(con_strict)
        constrained_rg_sum += con_rg
        constrained_graded_sum += con_reward
        constrained_rows.append(
            {
                "idx": idx,
                "target": entry["metadata"]["target"],
                "numbers": entry["metadata"]["numbers"],
                "gold": entry["answer"],
                "constrained": con_text,
                "rg_score": con_rg,
                "strict": con_strict,
                "graded_reward": con_reward,
                "denoise_forwards": constrained.denoise_forwards,
                "cache_read_calls": constrained.cache_read_calls,
                "cache_advance_calls": constrained.cache_advance_calls,
            }
        )

    elapsed = time.perf_counter() - started
    n = len(entries)
    return {
        "examples": n,
        "seconds": elapsed,
        "raw": {
            "strict_correct": raw_correct,
            "strict_accuracy": raw_correct / n if n else 0.0,
            "mean_reasoning_gym_score": raw_rg_sum / n if n else 0.0,
            "mean_graded_reward": raw_graded_sum / n if n else 0.0,
            "rows": raw_rows,
        },
        "constrained": {
            "strict_correct": constrained_correct,
            "strict_accuracy": constrained_correct / n if n else 0.0,
            "mean_reasoning_gym_score": constrained_rg_sum / n if n else 0.0,
            "mean_graded_reward": constrained_graded_sum / n if n else 0.0,
            "rows": constrained_rows,
        },
        "protected": {
            "strict_accuracy": None,
            "note": "No protected sidecar lane is used in this pilot.",
        },
        "raw_constrained_gap": (constrained_correct / n if n else 0.0) - (raw_correct / n if n else 0.0),
    }


def make_countdown_entries(args) -> tuple[Any, list[dict[str, Any]], Any, list[dict[str, Any]]]:
    import reasoning_gym.games.countdown  # noqa: F401
    from reasoning_gym.factory import create_dataset

    common = {
        "min_numbers": args.min_numbers,
        "max_numbers": args.max_numbers,
        "min_value": args.min_value,
        "max_value": args.max_value,
        "min_target": args.min_target,
        "max_target": args.max_target,
        "size": max(args.train_size, args.eval_size),
    }
    train_ds = create_dataset("countdown", seed=args.train_seed, **common)
    eval_ds = create_dataset("countdown", seed=args.eval_seed, **{**common, "size": args.eval_size})
    train_entries = [train_ds[idx] for idx in range(args.train_size)]
    eval_entries = [eval_ds[idx] for idx in range(args.eval_size)]
    return train_ds, train_entries, eval_ds, eval_entries


def resolve_stop_ids(config, tokenizer) -> list[int]:
    ids: list[int] = []

    def add(value) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        item = int(value)
        if item not in ids:
            ids.append(item)

    add(getattr(config, "eos_token_id", None))
    add(tokenizer.eos_token_id)
    for text in ("<|im_end|>", "<|endoftext|>"):
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            add(token_ids[0])
    if not ids:
        raise ValueError("Could not resolve any stop token ids")
    return ids


def load_model_and_tokenizer(args):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_model = Path(args.base_model)
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    adapter_in = Path(args.adapter_in) if args.adapter_in else None
    if adapter_in and (adapter_in / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
    else:
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    mask_id = getattr(config, "mask_token_id", None)
    if mask_id is None:
        mask_id = tokenizer.convert_tokens_to_ids("|<MASK>|")
    if mask_id is None or int(mask_id) < 0:
        raise ValueError("Could not resolve mask_token_id")
    stop_ids = resolve_stop_ids(config, tokenizer)
    return model, tokenizer, int(mask_id), stop_ids


def train(args) -> dict[str, Any]:
    configure_cuda_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Countdown RL pilot")
    torch.cuda.set_device(args.gpu_index)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    train_ds, train_entries, eval_ds, eval_entries = make_countdown_entries(args)
    model, tokenizer, mask_id, stop_ids = load_model_and_tokenizer(args)
    grammar = build_token_grammar(tokenizer, mask_id, stop_ids)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    config_payload = {
        "args": vars(args),
        "mask_id": mask_id,
        "stop_ids": stop_ids,
        "token_grammar": grammar.char_to_id,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "reward": "graded_countdown_exact_1_inverse_distance_partial",
        "policy_distribution": "model logits masked to Countdown grammar allowed token ids",
        "raw_rl_distribution": "raw full-vocab diffusion distribution with mask token banned",
        "serving_forward": "RequestDiffusionState cached route_i FLARE noisy forward"
        if args.use_fast_serving_cache
        else "cache_off_full_context_model_forward",
        "logprob_forward": "training forward exact re-score; serving logits are not differentiated",
        "lambda_raw": args.lambda_raw,
        "raw_rl_weight": args.raw_rl_weight,
        "raw_rl_group_size": args.raw_rl_group_size or args.group_size,
    }
    write_json(out_dir / "config.json", config_payload)

    print("[config] " + json.dumps(config_payload, sort_keys=True), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_all = time.perf_counter()

    eval_history: list[dict[str, Any]] = []
    if not args.skip_initial_eval:
        print("[eval] initial", flush=True)
        initial_eval = evaluate_lanes(model, tokenizer, eval_ds, eval_entries, grammar, args, generator)
        initial_eval["step"] = 0
        eval_history.append(initial_eval)
        write_json(out_dir / "eval_step_0000.json", initial_eval)
        print(
            "[eval] step=0 "
            f"raw={initial_eval['raw']['strict_accuracy']:.4f} "
            f"constrained={initial_eval['constrained']['strict_accuracy']:.4f}",
            flush=True,
        )

    step_rows: list[dict[str, Any]] = []
    with GpuMonitor(args.gpu_index, interval=args.gpu_monitor_interval) as monitor:
        for step in range(1, args.max_steps + 1):
            prompt_idx = (step - 1) % len(train_entries)
            entry = train_entries[prompt_idx]
            model.eval()
            rollout = constrained_countdown_rollout(
                model,
                tokenizer,
                train_ds,
                entry,
                prompt_idx,
                grammar,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.train_temperature,
                record_steps=True,
                use_fast_cache=args.use_fast_serving_cache,
                block_size=args.block_size,
                generator=generator,
            )
            rewards = rollout.rewards
            advantages = grpo_advantages(rewards)
            zero_advantage = bool(torch.count_nonzero(advantages).item() == 0)
            has_raw_internalization = bool(args.lambda_raw > 0 and any(item >= 1.0 - 1e-9 for item in rollout.strict_rewards))
            raw_rollout = None
            raw_advantages = None
            raw_zero_advantage = None
            if args.raw_rl_weight > 0:
                raw_rollout = raw_diffusion_rollout(
                    model,
                    tokenizer,
                    train_ds,
                    entry,
                    prompt_idx,
                    grammar,
                    group_size=args.raw_rl_group_size or args.group_size,
                    max_new_tokens=args.raw_rl_max_new_tokens or args.max_new_tokens,
                    threshold=args.raw_rl_threshold,
                    temperature=args.raw_rl_temperature,
                    top_p=args.raw_rl_top_p,
                    record_steps=True,
                    use_fast_cache=args.use_fast_serving_cache,
                    block_size=args.block_size,
                    generator=generator,
                )
                raw_advantages = grpo_advantages(raw_rollout.rewards)
                raw_zero_advantage = bool(torch.count_nonzero(raw_advantages).item() == 0)

            sync_cuda()
            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            model.train()
            did_backward = False
            if rollout_policy_token_count(rollout) and (not zero_advantage or has_raw_internalization):
                loss_metrics = backward_dual_term_loss(
                    model,
                    rollout,
                    advantages,
                    args.train_temperature,
                    lambda_raw=args.lambda_raw,
                    rescore_micro_batch_size=args.rescore_micro_batch_size,
                )
                did_backward = True
            else:
                loss_metrics = {
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "raw_internalize_loss": 0.0,
                    "policy_tokens": rollout_policy_token_count(rollout),
                    "raw_internalize_tokens": 0,
                    "raw_internalize_samples": 0,
                    "mean_logprob": None,
                    "mean_raw_logprob": None,
                }
            if raw_rollout is not None and raw_advantages is not None and raw_rollout_policy_token_count(raw_rollout):
                if not raw_zero_advantage:
                    raw_loss_metrics = backward_raw_rollout_policy_loss(
                        model,
                        raw_rollout,
                        raw_advantages,
                        temperature=args.raw_rl_temperature,
                        mask_id=mask_id,
                        raw_rl_weight=args.raw_rl_weight,
                        rescore_micro_batch_size=args.rescore_micro_batch_size,
                    )
                    did_backward = True
                else:
                    raw_loss_metrics = {
                        "raw_rl_loss": 0.0,
                        "raw_rl_tokens": raw_rollout_policy_token_count(raw_rollout),
                        "raw_rl_mean_logprob": None,
                    }
            else:
                raw_loss_metrics = {
                    "raw_rl_loss": 0.0,
                    "raw_rl_tokens": 0,
                    "raw_rl_mean_logprob": None,
                }
            if did_backward:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
                grad_norm_value = float(grad_norm.detach().float().cpu().item())
            else:
                grad_norm_value = 0.0
            optimizer.zero_grad(set_to_none=True)
            sync_cuda()
            update_seconds = time.perf_counter() - update_start
            total_loss = float(loss_metrics["loss"]) + float(args.raw_rl_weight) * float(raw_loss_metrics["raw_rl_loss"])
            raw_rollout_seconds = 0.0 if raw_rollout is None else float(raw_rollout.seconds)

            row = {
                "step": step,
                "prompt_idx": prompt_idx,
                "target": entry["metadata"]["target"],
                "numbers": entry["metadata"]["numbers"],
                "gold": entry["answer"],
                "expressions": rollout.expressions,
                "reasoning_gym_scores": rollout.rg_scores,
                "strict_rewards": rollout.strict_rewards,
                "graded_rewards": rewards,
                "reward_mean": float(sum(rewards) / len(rewards)) if rewards else 0.0,
                "reward_std": float(torch.tensor(rewards).std(unbiased=False).item()) if rewards else 0.0,
                "advantages": [float(item) for item in advantages.detach().cpu().tolist()],
                "zero_advantage": zero_advantage,
                "raw_internalization_active": has_raw_internalization,
                "rollout_seconds": rollout.seconds,
                "raw_rl_rollout_seconds": raw_rollout_seconds,
                "update_seconds": update_seconds,
                "step_seconds": rollout.seconds + raw_rollout_seconds + update_seconds,
                "denoise_forwards": rollout.denoise_forwards,
                "cache_read_calls": rollout.cache_read_calls,
                "cache_advance_calls": rollout.cache_advance_calls,
                "grad_norm": grad_norm_value,
                **loss_metrics,
                "loss_total": total_loss,
                "raw_rl_enabled": raw_rollout is not None,
                "raw_rl_texts": [] if raw_rollout is None else raw_rollout.texts,
                "raw_rl_reasoning_gym_scores": [] if raw_rollout is None else raw_rollout.rg_scores,
                "raw_rl_strict_rewards": [] if raw_rollout is None else raw_rollout.strict_rewards,
                "raw_rl_graded_rewards": [] if raw_rollout is None else raw_rollout.rewards,
                "raw_rl_reward_mean": 0.0
                if raw_rollout is None or not raw_rollout.rewards
                else float(sum(raw_rollout.rewards) / len(raw_rollout.rewards)),
                "raw_rl_reward_std": None
                if raw_rollout is None
                else float(torch.tensor(raw_rollout.rewards).std(unbiased=False).item()),
                "raw_rl_advantages": []
                if raw_advantages is None
                else [float(item) for item in raw_advantages.detach().cpu().tolist()],
                "raw_rl_zero_advantage": raw_zero_advantage,
                "raw_rl_denoise_forwards": 0 if raw_rollout is None else raw_rollout.denoise_forwards,
                "raw_rl_cache_read_calls": 0 if raw_rollout is None else raw_rollout.cache_read_calls,
                "raw_rl_cache_advance_calls": 0 if raw_rollout is None else raw_rollout.cache_advance_calls,
                **raw_loss_metrics,
            }
            step_rows.append(row)
            append_jsonl(metrics_path, row)
            if step == 1 or step % args.log_every == 0:
                print(
                    "[train] "
                    f"step={step} reward={row['reward_mean']:.3f} "
                    f"raw_reward={row['raw_rl_reward_mean']:.3f} "
                    f"loss={row['loss_total']:.4g} raw_ce_tok={row['raw_internalize_tokens']} "
                    f"raw_rl_tok={row['raw_rl_tokens']} "
                    f"rollout_s={row['rollout_seconds']:.2f} update_s={row['update_seconds']:.2f} "
                    f"zero_adv={zero_advantage} raw_zero_adv={raw_zero_advantage}",
                    flush=True,
                )

            if args.eval_every > 0 and step % args.eval_every == 0:
                print(f"[eval] step={step}", flush=True)
                eval_result = evaluate_lanes(model, tokenizer, eval_ds, eval_entries, grammar, args, generator)
                eval_result["step"] = step
                eval_history.append(eval_result)
                write_json(out_dir / f"eval_step_{step:04d}.json", eval_result)
                print(
                    "[eval] "
                    f"step={step} raw={eval_result['raw']['strict_accuracy']:.4f} "
                    f"constrained={eval_result['constrained']['strict_accuracy']:.4f}",
                    flush=True,
                )

    total_seconds = time.perf_counter() - started_all
    gpu_summary = monitor.summary()
    cuda_peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    cuda_peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

    if not eval_history or eval_history[-1].get("step") != args.max_steps:
        print("[eval] final", flush=True)
        final_eval = evaluate_lanes(model, tokenizer, eval_ds, eval_entries, grammar, args, generator)
        final_eval["step"] = args.max_steps
        eval_history.append(final_eval)
        write_json(out_dir / f"eval_step_{args.max_steps:04d}.json", final_eval)

    adapter_out = out_dir / "adapter_model"
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(out_dir)

    avg_step_seconds = (
        sum(row["step_seconds"] for row in step_rows) / len(step_rows)
        if step_rows
        else None
    )
    avg_rollout_seconds = (
        sum(row["rollout_seconds"] for row in step_rows) / len(step_rows)
        if step_rows
        else None
    )
    avg_update_seconds = (
        sum(row["update_seconds"] for row in step_rows) / len(step_rows)
        if step_rows
        else None
    )
    total_policy_tokens = sum(int(row["policy_tokens"]) for row in step_rows)
    total_rollout_seconds = sum(float(row["rollout_seconds"]) for row in step_rows)
    total_raw_rl_tokens = sum(int(row["raw_rl_tokens"]) for row in step_rows)
    total_raw_rl_rollout_seconds = sum(float(row["raw_rl_rollout_seconds"]) for row in step_rows)
    total_step_seconds = sum(float(row["step_seconds"]) for row in step_rows)
    total_raw_internalize_tokens = sum(int(row["raw_internalize_tokens"]) for row in step_rows)
    nonzero_advantage_steps = sum(int(not row["zero_advantage"]) for row in step_rows)
    raw_internalization_steps = sum(int(row["raw_internalization_active"]) for row in step_rows)
    raw_rl_rows = [row for row in step_rows if row["raw_rl_enabled"]]
    raw_rl_zero_advantage_steps = sum(int(bool(row["raw_rl_zero_advantage"])) for row in raw_rl_rows)
    summary = {
        "output_dir": str(out_dir),
        "adapter_out": str(adapter_out),
        "steps": len(step_rows),
        "total_seconds": total_seconds,
        "avg_step_seconds": avg_step_seconds,
        "avg_rollout_seconds": avg_rollout_seconds,
        "avg_update_seconds": avg_update_seconds,
        "total_policy_tokens": total_policy_tokens,
        "total_raw_rl_tokens": total_raw_rl_tokens,
        "total_all_policy_tokens": total_policy_tokens + total_raw_rl_tokens,
        "train_tokens_per_second": total_policy_tokens / total_step_seconds if total_step_seconds > 0 else None,
        "train_all_policy_tokens_per_second": (total_policy_tokens + total_raw_rl_tokens) / total_step_seconds if total_step_seconds > 0 else None,
        "rollout_tokens_per_second": total_policy_tokens / total_rollout_seconds if total_rollout_seconds > 0 else None,
        "raw_rl_rollout_tokens_per_second": total_raw_rl_tokens / total_raw_rl_rollout_seconds if total_raw_rl_rollout_seconds > 0 else None,
        "avg_raw_rl_rollout_seconds": total_raw_rl_rollout_seconds / len(raw_rl_rows) if raw_rl_rows else 0.0,
        "total_raw_internalize_tokens": total_raw_internalize_tokens,
        "raw_internalization_steps": raw_internalization_steps,
        "cuda_peak_allocated_gb": cuda_peak_allocated_gb,
        "cuda_peak_reserved_gb": cuda_peak_reserved_gb,
        "gpu": gpu_summary,
        "train_reward_mean_last": step_rows[-1]["reward_mean"] if step_rows else None,
        "zero_advantage_steps": sum(int(row["zero_advantage"]) for row in step_rows),
        "zero_advantage_rate": (sum(int(row["zero_advantage"]) for row in step_rows) / len(step_rows)) if step_rows else None,
        "nonzero_advantage_steps": nonzero_advantage_steps,
        "use_fast_serving_cache": bool(args.use_fast_serving_cache),
        "block_size": int(args.block_size),
        "lambda_raw": float(args.lambda_raw),
        "raw_rl_weight": float(args.raw_rl_weight),
        "raw_rl_zero_advantage_steps": raw_rl_zero_advantage_steps,
        "raw_rl_zero_advantage_rate": raw_rl_zero_advantage_steps / len(raw_rl_rows) if raw_rl_rows else None,
        "raw_rl_nonzero_advantage_steps": len(raw_rl_rows) - raw_rl_zero_advantage_steps,
        "raw_rl_group_size": int(args.raw_rl_group_size or args.group_size),
        "rescore_micro_batch_size": int(args.rescore_micro_batch_size),
        "eval_history": [
            {
                "step": item["step"],
                "raw_strict_accuracy": item["raw"]["strict_accuracy"],
                "constrained_strict_accuracy": item["constrained"]["strict_accuracy"],
                "raw_constrained_gap": item["raw_constrained_gap"],
                "raw_mean_reasoning_gym_score": item["raw"]["mean_reasoning_gym_score"],
                "constrained_mean_reasoning_gym_score": item["constrained"]["mean_reasoning_gym_score"],
                "raw_mean_graded_reward": item["raw"]["mean_graded_reward"],
                "constrained_mean_graded_reward": item["constrained"]["mean_graded_reward"],
            }
            for item in eval_history
        ],
    }
    write_json(out_dir / "summary.json", summary)
    print("[summary] " + json.dumps(summary, sort_keys=True), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE))
    parser.add_argument("--adapter-in", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--train-seed", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--min-numbers", type=int, default=4)
    parser.add_argument("--max-numbers", type=int, default=4)
    parser.add_argument("--min-value", type=int, default=1)
    parser.add_argument("--max-value", type=int, default=20)
    parser.add_argument("--min-target", type=int, default=10)
    parser.add_argument("--max-target", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument("--eval-constrained-temperature", type=float, default=0.0)
    parser.add_argument("--raw-temperature", type=float, default=0.0)
    parser.add_argument("--raw-top-p", type=float, default=0.95)
    parser.add_argument("--raw-threshold", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-raw", type=float, default=0.5)
    parser.add_argument("--raw-rl-weight", type=float, default=0.0)
    parser.add_argument("--raw-rl-group-size", type=int, default=0)
    parser.add_argument("--raw-rl-max-new-tokens", type=int, default=0)
    parser.add_argument("--raw-rl-temperature", type=float, default=1.0)
    parser.add_argument("--raw-rl-top-p", type=float, default=1.0)
    parser.add_argument("--raw-rl-threshold", type=float, default=0.3)
    parser.add_argument("--rescore-micro-batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--use-fast-serving-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-monitor-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
