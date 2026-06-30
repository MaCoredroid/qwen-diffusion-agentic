from dataclasses import dataclass
import json
import os
import time
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration import Fast_dLLM_Qwen3_5Config


FAST_DLLM_QWEN3_5_BRIDGE_STATUS = "implemented"
FAST_DLLM_QWEN3_5_GDN_MODE = "option_a_causal_gdn_v0"
FAST_DLLM_QWEN3_5_GDN_ENV = "FAST_DLLM_QWEN3_5_GDN_MODE"
FAST_DLLM_TRAIN_BD_SIZE_ENV = "FASTDLLM_TRAIN_BD_SIZE"
FAST_DLLM_TRAIN_BD_SIZE_CHOICES_ENV = "FASTDLLM_TRAIN_BD_SIZE_CHOICES"
FAST_DLLM_FLARE_TWO_STREAM_ENV = "FASTDLLM_FLARE_TWO_STREAM"
FLARE_TWO_STREAM_ENV = "FLARE_TWO_STREAM"
FASTDLLM_FLARE_GDN_ROUTE_ENV = "FASTDLLM_FLARE_GDN_ROUTE"
FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS_ENV = "FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS"
FASTDLLM_FLARE_ROUTE_II_CHECKPOINT_ENV = "FASTDLLM_FLARE_ROUTE_II_CHECKPOINT"
FASTDLLM_FLARE_ROUTE_II_BUG_ENV = "FASTDLLM_FLARE_ROUTE_II_BUG"
FASTDLLM_PROFILE_GDN_SCAN_ENV = "FASTDLLM_PROFILE_GDN_SCAN"
FASTDLLM_PROFILE_FLARE_SECTIONS_ENV = "FASTDLLM_PROFILE_FLARE_SECTIONS"
FASTDLLM_COMPILE_GDN_SCAN_ENV = "FASTDLLM_COMPILE_GDN_SCAN"
FASTDLLM_COMPILE_GDN_SCAN_MODE_ENV = "FASTDLLM_COMPILE_GDN_SCAN_MODE"
FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN_ENV = "FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN"
FASTDLLM_BATCH_FLARE_NOISY_GDN_ENV = "FASTDLLM_BATCH_FLARE_NOISY_GDN"
FASTDLLM_GDN_KERNEL_ENV = "FASTDLLM_GDN_KERNEL"
IGNORE_INDEX = -100
_FASTDLLM_GDN_PROFILE = None
_FASTDLLM_FLARE_SECTION_PROFILE = None
_FASTDLLM_COMPILED_GDN_RULES = {}


@dataclass
class Fast_dLLM_Qwen3_5ModelOutputWithPast(BaseModelOutputWithPast):
    block_past_key_values: Optional[Cache] = None


@dataclass
class Fast_dLLM_Qwen3_5CausalLMOutputWithPast(CausalLMOutputWithPast):
    block_past_key_values: Optional[Cache] = None


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def block_diff_mask(seq_len, block_size, device):
    idx = torch.arange(seq_len * 2, device=device)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]
    q_x0 = q_idx >= seq_len
    kv_x0 = kv_idx >= seq_len
    block_q = torch.where(q_x0, (q_idx - seq_len) // block_size, q_idx // block_size)
    block_kv = torch.where(kv_x0, (kv_idx - seq_len) // block_size, kv_idx // block_size)
    block_diagonal = (block_q == block_kv) & (q_x0 == kv_x0)
    offset_block_causal = (block_q > block_kv) & kv_x0 & ~q_x0
    block_causal = (block_q >= block_kv) & kv_x0 & q_x0
    return block_diagonal | offset_block_causal | block_causal


def causal_bool_mask(seq_len, device):
    idx = torch.arange(seq_len, device=device)
    return idx[:, None] >= idx[None, :]


def env_flag_enabled(*names):
    for name in names:
        raw = os.environ.get(name, "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
    return False


def env_flag_disabled(name):
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"0", "false", "no", "off"}


def flare_gdn_route():
    raw = os.environ.get(FASTDLLM_FLARE_GDN_ROUTE_ENV, "route_i").strip().lower()
    aliases = {
        "": "route_i",
        "i": "route_i",
        "1": "route_i",
        "route1": "route_i",
        "route_i": "route_i",
        "route-i": "route_i",
        "ii": "route_ii",
        "2": "route_ii",
        "route2": "route_ii",
        "route_ii": "route_ii",
        "route-ii": "route_ii",
    }
    if raw not in aliases:
        raise ValueError(f"Unsupported {FASTDLLM_FLARE_GDN_ROUTE_ENV}={raw!r}")
    return aliases[raw]


def flare_route_ii_stride_blocks():
    raw = os.environ.get(FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS_ENV, "8").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{FASTDLLM_FLARE_ROUTE_II_STRIDE_BLOCKS_ENV} must be a positive integer")
    return value


def flare_route_ii_bug_mode():
    return os.environ.get(FASTDLLM_FLARE_ROUTE_II_BUG_ENV, "").strip().lower()


def should_checkpoint_route_ii(*tensors):
    if env_flag_disabled(FASTDLLM_FLARE_ROUTE_II_CHECKPOINT_ENV):
        return False
    return torch.is_grad_enabled() and any(torch.is_tensor(tensor) and tensor.requires_grad for tensor in tensors)


def local_position_ids_from_doc_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    valid = doc_ids >= 0
    prev_valid = F.pad(valid[:, :-1], (1, 0), value=False)
    prev_doc_ids = F.pad(doc_ids[:, :-1], (1, 0), value=-1)
    segment_start = valid & (~prev_valid | (doc_ids != prev_doc_ids))
    positions = torch.arange(doc_ids.shape[1], device=doc_ids.device, dtype=doc_ids.dtype).unsqueeze(0)
    start_positions = torch.where(segment_start, positions, torch.zeros_like(positions))
    last_start_positions = torch.cummax(start_positions, dim=1).values
    return torch.where(valid, positions - last_start_positions, torch.zeros_like(doc_ids))


def contiguous_doc_segments(doc_ids_row: torch.Tensor):
    segments = []
    length = doc_ids_row.numel()
    index = 0
    while index < length:
        doc_id = int(doc_ids_row[index].item())
        if doc_id < 0:
            index += 1
            continue
        end = index + 1
        while end < length and int(doc_ids_row[end].item()) == doc_id:
            end += 1
        segments.append((index, end, doc_id))
        index = end
    return segments


def doc_causal_bool_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    seq_len = doc_ids.shape[1]
    local_pos = local_position_ids_from_doc_ids(doc_ids)
    valid = doc_ids >= 0
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    causal = local_pos[:, None, :] <= local_pos[:, :, None]
    mask = valid[:, :, None] & valid[:, None, :] & same_doc & causal
    eye = torch.eye(seq_len, dtype=torch.bool, device=doc_ids.device).unsqueeze(0)
    mask = mask | (~valid[:, :, None] & eye)
    return mask[:, None, :, :]


def flare_two_stream_bool_mask(doc_ids: torch.Tensor, block_size: int) -> torch.Tensor:
    """Build [clean, noisy] FLARE attention mask with packed-doc isolation."""
    batch_size, seq_len = doc_ids.shape
    total_len = seq_len * 2
    local_pos = local_position_ids_from_doc_ids(doc_ids)
    local_block = torch.div(local_pos, block_size, rounding_mode="floor")
    stream_index = torch.arange(total_len, device=doc_ids.device)
    stream_clean = stream_index < seq_len
    stream_pos = stream_index.remainder(seq_len)
    stream_doc_ids = doc_ids[:, stream_pos]
    stream_local_pos = local_pos[:, stream_pos]
    stream_local_block = local_block[:, stream_pos]
    stream_valid = stream_doc_ids >= 0

    q_clean = stream_clean[:, None]
    kv_clean = stream_clean[None, :]
    same_doc = stream_doc_ids[:, :, None] == stream_doc_ids[:, None, :]
    valid_pair = stream_valid[:, :, None] & stream_valid[:, None, :] & same_doc
    kv_causal_to_q = stream_local_pos[:, None, :] <= stream_local_pos[:, :, None]
    kv_before_q_block = stream_local_pos[:, None, :] < (stream_local_block[:, :, None] * block_size)
    same_block = stream_local_block[:, :, None] == stream_local_block[:, None, :]

    clean_query_mask = q_clean & kv_clean & kv_causal_to_q
    noisy_to_clean_mask = ~q_clean & kv_clean & kv_before_q_block
    noisy_to_noisy_mask = ~q_clean & ~kv_clean & same_block
    mask = valid_pair & (clean_query_mask | noisy_to_clean_mask | noisy_to_noisy_mask)
    eye = torch.eye(total_len, dtype=torch.bool, device=doc_ids.device).unsqueeze(0)
    mask = mask | (~stream_valid[:, :, None] & eye)
    return mask[:, None, :, :]


def parse_positive_int_list(raw, env_name):
    values = []
    for item in raw.replace(";", ",").replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {env_name} item={item!r}") from exc
        if value <= 0:
            raise ValueError(f"{env_name} values must be positive")
        values.append(value)
    return tuple(values)


class Fast_dLLM_Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Fast_dLLM_Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Fast_dLLM_Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Fast_dLLM_Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        rope = config.rope_parameters or {}
        base = rope.get("rope_theta", 10000000)
        partial_rotary_factor = rope.get("partial_rotary_factor", 0.25)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _gdn_profile_enabled():
    return env_flag_enabled(FASTDLLM_PROFILE_GDN_SCAN_ENV)


def _flare_section_profile_enabled():
    return env_flag_enabled(FASTDLLM_PROFILE_FLARE_SECTIONS_ENV)


def _optimize_flare_clean_gdn_enabled():
    return env_flag_enabled(FASTDLLM_OPTIMIZE_FLARE_CLEAN_GDN_ENV)


def _batch_flare_noisy_gdn_enabled():
    return env_flag_enabled(FASTDLLM_BATCH_FLARE_NOISY_GDN_ENV)


def _gdn_compile_enabled():
    return env_flag_enabled(FASTDLLM_COMPILE_GDN_SCAN_ENV)


def _gdn_kernel_backend():
    return os.environ.get(FASTDLLM_GDN_KERNEL_ENV, "torch").strip().lower() or "torch"


def _fla_gdn_kernel_enabled():
    return _gdn_kernel_backend() in {"fla", "flash-linear-attention", "flash_linear_attention"}


def _cuda_synchronize_for_profile(*tensors):
    if not torch.cuda.is_available():
        return
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
            return


def _reset_gdn_profile():
    global _FASTDLLM_GDN_PROFILE
    _FASTDLLM_GDN_PROFILE = {
        "scan_calls": 0,
        "scan_seconds": 0.0,
        "scan_shapes": {},
        "kernel_backend": _gdn_kernel_backend(),
        "compile_enabled": _gdn_compile_enabled(),
        "compile_mode": os.environ.get(FASTDLLM_COMPILE_GDN_SCAN_MODE_ENV, "default").strip() or "default",
    }


def _reset_flare_section_profile():
    global _FASTDLLM_FLARE_SECTION_PROFILE
    _FASTDLLM_FLARE_SECTION_PROFILE = {
        "sections": {},
        "clean_gdn_optimized": _optimize_flare_clean_gdn_enabled(),
    }


def _snapshot_gdn_profile():
    if _FASTDLLM_GDN_PROFILE is None:
        return None
    payload = dict(_FASTDLLM_GDN_PROFILE)
    payload["scan_shapes"] = dict(payload.get("scan_shapes", {}))
    return payload


def _snapshot_flare_section_profile():
    if _FASTDLLM_FLARE_SECTION_PROFILE is None:
        return None
    payload = dict(_FASTDLLM_FLARE_SECTION_PROFILE)
    payload["sections"] = {
        name: dict(values) for name, values in payload.get("sections", {}).items()
    }
    return payload


def _record_gdn_scan_profile(query, chunk_size, output_final_state, output_chunk_states, elapsed):
    if _FASTDLLM_GDN_PROFILE is None:
        return
    _FASTDLLM_GDN_PROFILE["scan_calls"] += 1
    _FASTDLLM_GDN_PROFILE["scan_seconds"] += float(elapsed)
    shape_key = (
        f"shape={tuple(query.shape)} chunk={int(chunk_size)} "
        f"final={bool(output_final_state)} chunks={bool(output_chunk_states)}"
    )
    shapes = _FASTDLLM_GDN_PROFILE["scan_shapes"]
    entry = shapes.setdefault(shape_key, {"calls": 0, "seconds": 0.0})
    entry["calls"] += 1
    entry["seconds"] += float(elapsed)


def _record_flare_section(name, elapsed):
    if _FASTDLLM_FLARE_SECTION_PROFILE is None:
        return
    sections = _FASTDLLM_FLARE_SECTION_PROFILE["sections"]
    entry = sections.setdefault(name, {"calls": 0, "seconds": 0.0})
    entry["calls"] += 1
    entry["seconds"] += float(elapsed)


def _time_flare_section(name, fn, *sync_tensors):
    if not _flare_section_profile_enabled():
        return fn()
    _cuda_synchronize_for_profile(*sync_tensors)
    start = time.time()
    result = fn()
    if isinstance(result, tuple):
        _cuda_synchronize_for_profile(*(item for item in result if torch.is_tensor(item)))
    elif torch.is_tensor(result):
        _cuda_synchronize_for_profile(result)
    else:
        _cuda_synchronize_for_profile(*sync_tensors)
    _record_flare_section(name, time.time() - start)
    return result


def _compiled_gdn_rule():
    mode = os.environ.get(FASTDLLM_COMPILE_GDN_SCAN_MODE_ENV, "default").strip() or "default"
    key = mode
    compiled = _FASTDLLM_COMPILED_GDN_RULES.get(key)
    if compiled is not None:
        return compiled
    kwargs = {}
    if mode != "default":
        kwargs["mode"] = mode
    compiled = torch.compile(_torch_chunk_gated_delta_rule_impl, **kwargs)
    _FASTDLLM_COMPILED_GDN_RULES[key] = compiled
    print(f"[fastdllm-gdn-compile] enabled mode={mode}", flush=True)
    return compiled


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    output_chunk_states=False,
):
    if _fla_gdn_kernel_enabled() and not output_chunk_states:
        fn = _fla_chunk_gated_delta_rule_adapter
    else:
        fn = _compiled_gdn_rule() if _gdn_compile_enabled() else _torch_chunk_gated_delta_rule_impl
    if not _gdn_profile_enabled():
        return fn(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_chunk_states=output_chunk_states,
        )

    _cuda_synchronize_for_profile(query, key, value, g, beta)
    start = time.time()
    result = fn(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_chunk_states=output_chunk_states,
    )
    _cuda_synchronize_for_profile(query, key, value, g, beta)
    _record_gdn_scan_profile(query, chunk_size, output_final_state, output_chunk_states, time.time() - start)
    return result


def _fla_chunk_gated_delta_rule_adapter(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    output_chunk_states=False,
):
    if output_chunk_states:
        raise ValueError("FLA GDN adapter does not provide output_chunk_states")
    if not query.is_cuda:
        raise RuntimeError("FASTDLLM_GDN_KERNEL=fla requires CUDA tensors")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    except Exception as exc:
        raise RuntimeError(
            "FASTDLLM_GDN_KERNEL=fla requires flash-linear-attention to be installed"
        ) from exc

    output, final_state = fla_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        scale=query.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
    )
    return output, final_state


def _torch_chunk_gated_delta_rule_impl(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    output_chunk_states=False,
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    effective_chunks = (sequence_length + chunk_size - 1) // chunk_size
    chunk_states = [] if output_chunk_states else None
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        attn = attn.masked_fill(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
        if output_chunk_states:
            chunk_states.append(last_recurrent_state)
    if output_chunk_states:
        chunk_states = torch.stack(chunk_states, dim=1)[:, :effective_chunks]
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    if output_chunk_states:
        return core_attn_out, last_recurrent_state, chunk_states
    return core_attn_out, last_recurrent_state


class Fast_dLLM_Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Fast_dLLM_Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = Fast_dLLM_Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        output_chunk_states: bool = False,
        chunk_size: int = 64,
    ):
        if attention_mask is not None and attention_mask.dim() == 2:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        delta_outputs = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_chunk_states=output_chunk_states,
        )
        if output_chunk_states:
            core_attn_out, final_state, chunk_states = delta_outputs
        else:
            core_attn_out, final_state = delta_outputs
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
        if output_chunk_states:
            return output, final_state, chunk_states
        if output_final_state:
            return output, final_state
        return output


def gdn_project_and_conv(gdn_layer, hidden_states: torch.Tensor, conv_tail: Optional[torch.Tensor] = None):
    seq_len = hidden_states.shape[1]
    raw_qkv = gdn_layer.in_proj_qkv(hidden_states)
    if conv_tail is None or conv_tail.numel() == 0:
        conv_input = raw_qkv
        slice_start = 0
    else:
        lag = int(gdn_layer.conv_kernel_size) - 1
        if conv_tail.shape[1] < lag:
            pad = torch.zeros(
                conv_tail.shape[0],
                lag - conv_tail.shape[1],
                conv_tail.shape[2],
                dtype=conv_tail.dtype,
                device=conv_tail.device,
            )
            conv_tail = torch.cat([pad, conv_tail], dim=1)
        conv_tail = conv_tail[:, -lag:]
        conv_input = torch.cat([conv_tail, raw_qkv], dim=1)
        slice_start = conv_tail.shape[1]

    conv_all = gdn_layer.conv1d(conv_input.transpose(1, 2))
    conv_all = F.silu(conv_all[:, :, : conv_input.shape[1]]).transpose(1, 2)
    mixed_qkv = conv_all[:, slice_start : slice_start + seq_len]
    return raw_qkv, mixed_qkv


def run_gdn_manual_route_i(
    gdn_layer,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
    conv_tail: Optional[torch.Tensor] = None,
    output_chunk_states: bool = False,
):
    batch_size, seq_len, _ = hidden_states.shape
    raw_qkv, mixed_qkv = gdn_project_and_conv(gdn_layer, hidden_states, conv_tail=conv_tail)
    query, key, value = torch.split(
        mixed_qkv,
        [gdn_layer.key_dim, gdn_layer.key_dim, gdn_layer.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, gdn_layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, gdn_layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, gdn_layer.head_v_dim)
    z = gdn_layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, gdn_layer.head_v_dim)
    beta = gdn_layer.in_proj_b(hidden_states).sigmoid()
    g = -gdn_layer.A_log.float().exp() * F.softplus(
        gdn_layer.in_proj_a(hidden_states).float() + gdn_layer.dt_bias
    )

    if gdn_layer.num_v_heads // gdn_layer.num_k_heads > 1:
        repeat = gdn_layer.num_v_heads // gdn_layer.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    delta_outputs = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        output_chunk_states=output_chunk_states,
    )
    if output_chunk_states:
        core_attn_out, final_state, chunk_states = delta_outputs
    else:
        core_attn_out, final_state = delta_outputs
        chunk_states = None
    core_attn_out = gdn_layer.norm(
        core_attn_out.reshape(-1, gdn_layer.head_v_dim),
        z.reshape(-1, gdn_layer.head_v_dim),
    )
    output = gdn_layer.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
    return output, final_state, chunk_states, raw_qkv


def clean_gdn_docwise_with_boundaries(gdn_layer, clean_states: torch.Tensor, doc_ids: torch.Tensor, block_size: int):
    batch_size, seq_len, _ = clean_states.shape
    clean_output = torch.zeros_like(clean_states)
    clean_raw_qkv = torch.zeros(
        batch_size,
        seq_len,
        gdn_layer.conv_dim,
        dtype=clean_states.dtype,
        device=clean_states.device,
    )
    boundary_states = {}

    for batch in range(batch_size):
        for start, end, _ in contiguous_doc_segments(doc_ids[batch]):
            segment = clean_states[batch : batch + 1, start:end]
            if _optimize_flare_clean_gdn_enabled():
                segment_output, _, chunk_states, raw_qkv = run_gdn_manual_route_i(
                    gdn_layer,
                    segment,
                    chunk_size=block_size,
                    output_chunk_states=True,
                )
                clean_output[batch : batch + 1, start:end] = segment_output
            else:
                clean_output[batch : batch + 1, start:end] = gdn_layer(segment)
                _, _, chunk_states, raw_qkv = run_gdn_manual_route_i(
                    gdn_layer,
                    segment,
                    chunk_size=block_size,
                    output_chunk_states=True,
                )
            clean_raw_qkv[batch : batch + 1, start:end] = raw_qkv
            if chunk_states is None:
                raise RuntimeError("GDN Route-I clean pass did not return chunk states")
            zero_state = torch.zeros_like(chunk_states[:, 0])
            num_blocks = (end - start + block_size - 1) // block_size
            for block_index in range(num_blocks):
                block_start = start + block_index * block_size
                if block_index == 0:
                    boundary_states[(batch, block_start)] = zero_state
                else:
                    boundary_states[(batch, block_start)] = chunk_states[:, block_index - 1]

    return clean_output, boundary_states, clean_raw_qkv


def noisy_gdn_route_i(
    gdn_layer,
    noisy_states: torch.Tensor,
    noisy_doc_ids: torch.Tensor,
    clean_doc_ids: torch.Tensor,
    clean_boundary_states,
    clean_raw_qkv: torch.Tensor,
    block_size: int,
):
    if _batch_flare_noisy_gdn_enabled():
        return noisy_gdn_route_i_batched(
            gdn_layer,
            noisy_states,
            noisy_doc_ids,
            clean_doc_ids,
            clean_boundary_states,
            clean_raw_qkv,
            block_size,
        )

    batch_size = clean_doc_ids.shape[0]
    noisy_output = torch.zeros_like(noisy_states)
    conv_lag = int(gdn_layer.conv_kernel_size) - 1

    for noisy_batch in range(noisy_states.shape[0]):
        clean_batch = noisy_batch % batch_size
        for start, end, _ in contiguous_doc_segments(noisy_doc_ids[noisy_batch]):
            for block_start in range(start, end, block_size):
                block_end = min(block_start + block_size, end)
                initial_state = clean_boundary_states[(clean_batch, block_start)]
                tail_start = max(start, block_start - conv_lag)
                conv_tail = clean_raw_qkv[clean_batch : clean_batch + 1, tail_start:block_start]
                if conv_tail.numel() == 0:
                    conv_tail = None
                block_output, _, _, _ = run_gdn_manual_route_i(
                    gdn_layer,
                    noisy_states[noisy_batch : noisy_batch + 1, block_start:block_end],
                    chunk_size=block_size,
                    initial_state=initial_state,
                    conv_tail=conv_tail,
                )
                noisy_output[noisy_batch : noisy_batch + 1, block_start:block_end] = block_output
    return noisy_output


def _gdn_conv_tail_for_block(
    *,
    clean_raw_qkv: torch.Tensor,
    clean_batch: int,
    doc_start: int,
    block_start: int,
    conv_lag: int,
):
    if conv_lag <= 0:
        return clean_raw_qkv.new_empty(0, clean_raw_qkv.shape[-1])
    tail = clean_raw_qkv[
        clean_batch,
        max(doc_start, block_start - conv_lag) : block_start,
    ]
    if tail.shape[0] >= conv_lag:
        return tail[-conv_lag:]
    padded = torch.zeros(
        conv_lag,
        clean_raw_qkv.shape[-1],
        dtype=clean_raw_qkv.dtype,
        device=clean_raw_qkv.device,
    )
    if tail.numel() > 0:
        padded[-tail.shape[0] :] = tail
    return padded


def noisy_gdn_route_i_batched(
    gdn_layer,
    noisy_states: torch.Tensor,
    noisy_doc_ids: torch.Tensor,
    clean_doc_ids: torch.Tensor,
    clean_boundary_states,
    clean_raw_qkv: torch.Tensor,
    block_size: int,
):
    batch_size = clean_doc_ids.shape[0]
    noisy_output = torch.zeros_like(noisy_states)
    conv_lag = int(gdn_layer.conv_kernel_size) - 1
    groups = {}

    for noisy_batch in range(noisy_states.shape[0]):
        clean_batch = noisy_batch % batch_size
        for doc_start, doc_end, _ in contiguous_doc_segments(noisy_doc_ids[noisy_batch]):
            for block_start in range(doc_start, doc_end, block_size):
                block_end = min(block_start + block_size, doc_end)
                block_len = int(block_end - block_start)
                groups.setdefault(block_len, []).append(
                    (
                        noisy_batch,
                        clean_batch,
                        int(doc_start),
                        int(block_start),
                        int(block_end),
                    )
                )

    for block_len, entries in groups.items():
        hidden_blocks = torch.cat(
            [
                noisy_states[noisy_batch : noisy_batch + 1, block_start:block_end]
                for noisy_batch, _, _, block_start, block_end in entries
            ],
            dim=0,
        )
        initial_states = torch.cat(
            [
                clean_boundary_states[(clean_batch, block_start)]
                for _, clean_batch, _, block_start, _ in entries
            ],
            dim=0,
        )
        if conv_lag > 0:
            conv_tail = torch.stack(
                [
                    _gdn_conv_tail_for_block(
                        clean_raw_qkv=clean_raw_qkv,
                        clean_batch=clean_batch,
                        doc_start=doc_start,
                        block_start=block_start,
                        conv_lag=conv_lag,
                    )
                    for _, clean_batch, doc_start, block_start, _ in entries
                ],
                dim=0,
            )
        else:
            conv_tail = None

        block_outputs, _, _, _ = run_gdn_manual_route_i(
            gdn_layer,
            hidden_blocks,
            chunk_size=block_size,
            initial_state=initial_states,
            conv_tail=conv_tail,
        )
        for idx, (noisy_batch, _, _, block_start, block_end) in enumerate(entries):
            noisy_output[noisy_batch : noisy_batch + 1, block_start:block_end] = block_outputs[idx : idx + 1]
    return noisy_output


def _empty_conv_tail(reference: torch.Tensor, gdn_layer):
    return reference.new_empty(reference.shape[0], 0, gdn_layer.conv_dim)


def _zero_gdn_state(reference: torch.Tensor, gdn_layer):
    return torch.zeros(
        reference.shape[0],
        gdn_layer.num_v_heads,
        gdn_layer.head_k_dim,
        gdn_layer.head_v_dim,
        dtype=torch.float32,
        device=reference.device,
    )


def _select_route_ii_conv_tail(
    *,
    running_tail: Optional[torch.Tensor],
    window_raw_qkv: torch.Tensor,
    doc_start: int,
    window_start: int,
    block_start: int,
    conv_lag: int,
):
    if conv_lag <= 0 or block_start <= doc_start:
        return window_raw_qkv.new_empty(window_raw_qkv.shape[0], 0, window_raw_qkv.shape[-1])

    tail_start = max(doc_start, block_start - conv_lag)
    pieces = []
    if tail_start < window_start:
        prior_needed = window_start - tail_start
        if running_tail is not None and running_tail.numel() > 0:
            pieces.append(running_tail[:, -prior_needed:])
    local_end = block_start - window_start
    if local_end > 0:
        local_start = max(0, tail_start - window_start)
        pieces.append(window_raw_qkv[:, local_start:local_end])
    if not pieces:
        return window_raw_qkv.new_empty(window_raw_qkv.shape[0], 0, window_raw_qkv.shape[-1])
    return torch.cat(pieces, dim=1)[:, -conv_lag:]


def clean_noisy_gdn_route_ii(
    gdn_layer,
    clean_states: torch.Tensor,
    noisy_states: torch.Tensor,
    noisy_doc_ids: torch.Tensor,
    clean_doc_ids: torch.Tensor,
    block_size: int,
    stride_blocks: int,
):
    batch_size, seq_len, _ = clean_states.shape
    clean_output = torch.zeros_like(clean_states)
    noisy_output = torch.zeros_like(noisy_states)
    conv_lag = int(gdn_layer.conv_kernel_size) - 1
    window_size = int(block_size) * int(stride_blocks)
    bug_mode = flare_route_ii_bug_mode()

    for batch in range(batch_size):
        carried_state = None
        carried_boundary_state = None
        carried_tail = None
        for doc_start, doc_end, _ in contiguous_doc_segments(clean_doc_ids[batch]):
            segment = clean_states[batch : batch + 1, doc_start:doc_end]
            clean_output[batch : batch + 1, doc_start:doc_end] = gdn_layer(segment)

            if bug_mode == "doc_reset":
                running_state = carried_state
                running_boundary_state = carried_boundary_state
                running_tail = carried_tail
            else:
                running_state = None
                running_boundary_state = None
                running_tail = None
            for window_start in range(doc_start, doc_end, window_size):
                window_end = min(window_start + window_size, doc_end)
                clean_window = clean_states[batch : batch + 1, window_start:window_end]
                state_arg = running_state
                tail_arg = running_tail

                def run_clean_window(window_hidden, initial_state, conv_tail):
                    _, final_state, chunk_states, raw_qkv = run_gdn_manual_route_i(
                        gdn_layer,
                        window_hidden,
                        chunk_size=block_size,
                        initial_state=initial_state,
                        conv_tail=conv_tail,
                        output_chunk_states=True,
                    )
                    if chunk_states is None:
                        raise RuntimeError("GDN Route-II clean window did not return chunk states")
                    return final_state, chunk_states, raw_qkv

                if should_checkpoint_route_ii(
                    clean_window,
                    state_arg if state_arg is not None else clean_window.new_empty(0),
                    tail_arg if tail_arg is not None else clean_window.new_empty(0),
                ):
                    if state_arg is None and tail_arg is None:
                        final_state, chunk_states, raw_qkv = torch_checkpoint(
                            lambda window_hidden: run_clean_window(window_hidden, None, None),
                            clean_window,
                            use_reentrant=False,
                        )
                    elif state_arg is None:
                        final_state, chunk_states, raw_qkv = torch_checkpoint(
                            lambda window_hidden, conv_tail: run_clean_window(window_hidden, None, conv_tail),
                            clean_window,
                            tail_arg,
                            use_reentrant=False,
                        )
                    elif tail_arg is None:
                        final_state, chunk_states, raw_qkv = torch_checkpoint(
                            lambda window_hidden, initial_state: run_clean_window(window_hidden, initial_state, None),
                            clean_window,
                            state_arg,
                            use_reentrant=False,
                        )
                    else:
                        final_state, chunk_states, raw_qkv = torch_checkpoint(
                            run_clean_window,
                            clean_window,
                            state_arg,
                            tail_arg,
                            use_reentrant=False,
                        )
                else:
                    final_state, chunk_states, raw_qkv = run_clean_window(clean_window, state_arg, tail_arg)
                raw_qkv_for_tail = torch.zeros_like(raw_qkv)
                raw_qkv_for_tail[:, :] = raw_qkv

                num_window_blocks = (window_end - window_start + block_size - 1) // block_size
                for noisy_batch in (batch, batch + batch_size):
                    if noisy_batch >= noisy_states.shape[0]:
                        continue
                    for block_offset in range(0, window_end - window_start, block_size):
                        block_start = window_start + block_offset
                        block_end = min(block_start + block_size, window_end)
                        block_index = block_offset // block_size
                        if bug_mode == "window_offset":
                            initial_state = chunk_states[:, min(block_index, chunk_states.shape[1] - 1)]
                        elif block_index == 0:
                            initial_state = running_boundary_state
                            if initial_state is None:
                                initial_state = torch.zeros_like(chunk_states[:, 0])
                        else:
                            initial_state = chunk_states[:, block_index - 1]
                        if bug_mode == "zero_seed" and block_start > doc_start:
                            initial_state = torch.zeros_like(initial_state)
                        conv_tail = _select_route_ii_conv_tail(
                            running_tail=tail_arg,
                            window_raw_qkv=raw_qkv_for_tail,
                            doc_start=doc_start,
                            window_start=window_start,
                            block_start=block_start,
                            conv_lag=conv_lag,
                        )
                        if bug_mode == "zero_tail":
                            conv_tail = None
                        elif conv_tail.numel() == 0:
                            conv_tail = None
                        block_output, _, _, _ = run_gdn_manual_route_i(
                            gdn_layer,
                            noisy_states[noisy_batch : noisy_batch + 1, block_start:block_end],
                            chunk_size=block_size,
                            initial_state=initial_state,
                            conv_tail=conv_tail,
                        )
                        noisy_output[noisy_batch : noisy_batch + 1, block_start:block_end] = block_output

                running_state = final_state
                running_boundary_state = chunk_states[:, -1]
                if conv_lag > 0:
                    if tail_arg is None or tail_arg.numel() == 0:
                        running_tail = raw_qkv_for_tail[:, -conv_lag:]
                    else:
                        running_tail = torch.cat([tail_arg, raw_qkv_for_tail], dim=1)[:, -conv_lag:]
                else:
                    running_tail = None

                if chunk_states.shape[1] != num_window_blocks:
                    raise RuntimeError(
                        f"GDN Route-II expected {num_window_blocks} chunk states, got {chunk_states.shape[1]}"
                    )

            carried_state = running_state
            carried_boundary_state = running_boundary_state
            carried_tail = running_tail

    return clean_output, noisy_output


class Fast_dLLM_Qwen3_5Attention(nn.Module):
    def __init__(self, config: Fast_dLLM_Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Fast_dLLM_Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Fast_dLLM_Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _project(self, hidden_states, position_embeddings, split_size=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if split_size is not None:
            q_1, q_2 = query_states[:, :, :split_size], query_states[:, :, split_size:]
            k_1, k_2 = key_states[:, :, :split_size], key_states[:, :, split_size:]
            q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)
            query_states = torch.cat([q_1, q_2], dim=2)
            key_states = torch.cat([k_1, k_2], dim=2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

    def forward(self, hidden_states, position_embeddings, attention_mask=None, split_size=None):
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states, gate = self._project(hidden_states, position_embeddings, split_size)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(attn_weights.dtype).min)
            else:
                attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Fast_dLLM_Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Fast_dLLM_Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.bd_size = config.bd_size
        self.gdn_mode = getattr(config, "gdn_mode", FAST_DLLM_QWEN3_5_GDN_MODE)
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Fast_dLLM_Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Fast_dLLM_Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer_type: {self.layer_type}")
        self.mlp = Fast_dLLM_Qwen3_5MLP(config)
        self.input_layernorm = Fast_dLLM_Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Fast_dLLM_Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _active_gdn_mode(self):
        return os.environ.get(FAST_DLLM_QWEN3_5_GDN_ENV, self.gdn_mode)

    def _linear_attn_by_blocks(self, hidden_states):
        block_size = int(self.bd_size)
        seq_len = hidden_states.shape[1]
        if seq_len % block_size:
            return self.linear_attn(hidden_states)
        batch_size, _, hidden_size = hidden_states.shape
        num_blocks = seq_len // block_size
        block_states = hidden_states.contiguous().view(batch_size, num_blocks, block_size, hidden_size)
        block_states = block_states.view(batch_size * num_blocks, block_size, hidden_size)
        block_outputs = self.linear_attn(block_states)
        return block_outputs.view(batch_size, num_blocks, block_size, hidden_size).reshape(batch_size, seq_len, hidden_size)

    def _linear_attn_clean_state_injection(self, noisy_states, clean_states):
        block_size = int(self.bd_size)
        seq_len = noisy_states.shape[1]
        if seq_len % block_size:
            return torch.cat([self.linear_attn(noisy_states), self.linear_attn(clean_states)], dim=1)
        batch_size, _, hidden_size = noisy_states.shape
        num_blocks = seq_len // block_size
        clean_states, _, clean_chunk_states = self.linear_attn(
            clean_states,
            output_final_state=True,
            output_chunk_states=True,
            chunk_size=block_size,
        )
        zero_state = torch.zeros_like(clean_chunk_states[:, :1])
        initial_states = torch.cat([zero_state, clean_chunk_states[:, :-1].detach()], dim=1)
        noisy_blocks = noisy_states.contiguous().view(batch_size, num_blocks, block_size, hidden_size)
        noisy_blocks = noisy_blocks.view(batch_size * num_blocks, block_size, hidden_size)
        state_shape = initial_states.shape[2:]
        initial_states = initial_states.reshape(batch_size * num_blocks, *state_shape)
        noisy_states = self.linear_attn(noisy_blocks, initial_state=initial_states, chunk_size=block_size)
        noisy_states = noisy_states.view(batch_size, num_blocks, block_size, hidden_size).reshape(batch_size, seq_len, hidden_size)
        return torch.cat([noisy_states, clean_states], dim=1)

    def _linear_attn_clean_state_dualpass(self, noisy_states, clean_states):
        block_size = int(self.bd_size)
        seq_len = noisy_states.shape[1]
        if seq_len % block_size:
            return self._linear_attn_clean_state_injection(noisy_states, clean_states)
        batch_size, _, hidden_size = noisy_states.shape
        num_blocks = seq_len // block_size
        clean_states, _, clean_chunk_states = self.linear_attn(
            clean_states,
            output_final_state=True,
            output_chunk_states=True,
            chunk_size=block_size,
        )
        zero_state = torch.zeros_like(clean_chunk_states[:, :1])
        initial_states = torch.cat([zero_state, clean_chunk_states[:, :-1].detach()], dim=1)
        noisy_blocks = noisy_states.contiguous().view(batch_size, num_blocks, block_size, hidden_size)
        noisy_blocks = noisy_blocks.view(batch_size * num_blocks, block_size, hidden_size)
        state_shape = initial_states.shape[2:]
        initial_states = initial_states.reshape(batch_size * num_blocks, *state_shape)
        forward_states = self.linear_attn(noisy_blocks, initial_state=initial_states, chunk_size=block_size)
        reverse_blocks = torch.flip(noisy_blocks, dims=[1])
        reverse_states = self.linear_attn(reverse_blocks, chunk_size=block_size)
        reverse_states = torch.flip(reverse_states, dims=[1])
        noisy_states = (forward_states + reverse_states) * 0.5
        noisy_states = noisy_states.view(batch_size, num_blocks, block_size, hidden_size).reshape(batch_size, seq_len, hidden_size)
        return torch.cat([noisy_states, clean_states], dim=1)

    def _linear_attn_mdm(self, hidden_states, mdm_split_size):
        noisy_states = hidden_states[:, :mdm_split_size]
        clean_states = hidden_states[:, mdm_split_size:]
        mode = self._active_gdn_mode()
        if mode in {"option_a_causal_gdn", "option_a_causal_gdn_v0"}:
            noisy_states = self.linear_attn(noisy_states)
        elif mode in {"option_a_noisy_block_isolation_v0", "clean_prefix_noisy_block_isolation_v0"}:
            noisy_states = self._linear_attn_by_blocks(noisy_states)
        elif mode in {"option_a_clean_state_injection_v0", "clean_state_injection_v0"}:
            return self._linear_attn_clean_state_injection(noisy_states, clean_states)
        elif mode in {"option_a_clean_state_dualpass_v0", "clean_state_dualpass_v0"}:
            return self._linear_attn_clean_state_dualpass(noisy_states, clean_states)
        else:
            raise ValueError(f"Unsupported Qwen3.5 GDN diffusion mode: {mode}")
        clean_states = self.linear_attn(clean_states)
        return torch.cat([noisy_states, clean_states], dim=1)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, mdm_split_size=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            if mdm_split_size is not None:
                hidden_states = self._linear_attn_mdm(hidden_states, mdm_split_size)
            else:
                hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                split_size=mdm_split_size,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Fast_dLLM_Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Fast_dLLM_Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Fast_dLLM_Qwen3_5RMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, Fast_dLLM_Qwen3_5RMSNormGated):
            module.weight.data.fill_(1.0)


class Fast_dLLM_Qwen3_5Model(Fast_dLLM_Qwen3_5PreTrainedModel):
    def __init__(self, config: Fast_dLLM_Qwen3_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _attention_mask(self, input_ids, labels, mdm_split_size):
        if self.training and mdm_split_size is not None:
            return block_diff_mask(mdm_split_size, self.bd_size, input_ids.device)[None, None, :, :]
        return causal_bool_mask(input_ids.shape[1], input_ids.device)[None, None, :, :]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        mdm_split_size: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if position_ids is None:
            pos_len = mdm_split_size if mdm_split_size is not None else inputs_embeds.shape[1]
            position_ids = torch.arange(pos_len, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        full_attention_mask = self._attention_mask(input_ids, labels, mdm_split_size).to(inputs_embeds.device)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        for layer in self.layers:
            layer_mask = full_attention_mask if layer.layer_type == "full_attention" else None
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    lambda states, current_layer=layer, current_mask=layer_mask: current_layer(
                        states,
                        position_embeddings=position_embeddings,
                        attention_mask=current_mask,
                        position_ids=position_ids,
                        mdm_split_size=mdm_split_size,
                    ),
                    hidden_states,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    mdm_split_size=mdm_split_size,
                )
        hidden_states = self.norm(hidden_states)
        return Fast_dLLM_Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Fast_dLLM_Qwen3_5ForCausalLM(Fast_dLLM_Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Fast_dLLM_Qwen3_5Config):
        super().__init__(config)
        self.model = Fast_dLLM_Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _set_active_train_bd_size(self, bd_size):
        bd_size = int(bd_size)
        if getattr(self, "_active_train_bd_size", None) == bd_size and int(self.model.bd_size) == bd_size:
            return
        for module in self.modules():
            if hasattr(module, "bd_size"):
                module.bd_size = bd_size
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "bd_size"):
                config.bd_size = bd_size
        self._active_train_bd_size = bd_size

    def _resolve_train_bd_size(self, seq_len, device):
        fixed_raw = os.environ.get(FAST_DLLM_TRAIN_BD_SIZE_ENV, "").strip()
        choices_raw = os.environ.get(FAST_DLLM_TRAIN_BD_SIZE_CHOICES_ENV, "").strip()
        if fixed_raw:
            choices = parse_positive_int_list(fixed_raw, FAST_DLLM_TRAIN_BD_SIZE_ENV)
            if len(choices) != 1:
                raise ValueError(f"{FAST_DLLM_TRAIN_BD_SIZE_ENV} accepts exactly one positive integer")
            bd_size = choices[0]
            if seq_len % bd_size:
                raise ValueError(f"input sequence length {seq_len} is not divisible by bd_size={bd_size}")
            return bd_size
        if not choices_raw:
            bd_size = int(self.model.bd_size)
            if seq_len % bd_size:
                raise ValueError(f"input sequence length {seq_len} is not divisible by bd_size={bd_size}")
            return bd_size
        choices = tuple(sorted(set(parse_positive_int_list(choices_raw, FAST_DLLM_TRAIN_BD_SIZE_CHOICES_ENV))))
        valid_choices = [bd_size for bd_size in choices if seq_len % bd_size == 0]
        if not valid_choices:
            raise ValueError(f"input sequence length {seq_len} is not divisible by any bd_size choice={choices}")
        index = int(torch.randint(len(valid_choices), (1,), device=device).item())
        return valid_choices[index]

    def _structural_loss_settings(self):
        weight_raw = os.environ.get("FASTDLLM_STRUCTURAL_LOSS_WEIGHT", "1.0")
        token_ids_raw = os.environ.get("FASTDLLM_STRUCTURAL_TOKEN_IDS", "")
        cache_key = (weight_raw, token_ids_raw)
        if getattr(self, "_structural_loss_cache_key", None) == cache_key:
            return self._structural_loss_cache

        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_STRUCTURAL_LOSS_WEIGHT={weight_raw!r}") from exc
        if weight <= 0:
            raise ValueError("FASTDLLM_STRUCTURAL_LOSS_WEIGHT must be positive")

        token_ids = []
        for item in token_ids_raw.replace(";", ",").replace(" ", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                token_ids.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid FASTDLLM_STRUCTURAL_TOKEN_IDS item={item!r}") from exc
        settings = (weight, tuple(sorted(set(token_ids))))
        self._structural_loss_cache_key = cache_key
        self._structural_loss_cache = settings
        return settings

    def _argument_span_loss_settings(self):
        weight_raw = os.environ.get("FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT", "1.0")
        start_ids_raw = os.environ.get("FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS", "")
        end_ids_raw = os.environ.get("FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS", "")
        cache_key = (weight_raw, start_ids_raw, end_ids_raw)
        if getattr(self, "_argument_span_loss_cache_key", None) == cache_key:
            return self._argument_span_loss_cache

        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT={weight_raw!r}") from exc
        if weight <= 0:
            raise ValueError("FASTDLLM_ARGUMENT_SPAN_LOSS_WEIGHT must be positive")

        def parse_ids(raw, env_name):
            token_ids = []
            for item in raw.replace(";", ",").replace(" ", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    token_ids.append(int(item))
                except ValueError as exc:
                    raise ValueError(f"Invalid {env_name} item={item!r}") from exc
            return tuple(sorted(set(token_ids)))

        settings = (
            weight,
            parse_ids(start_ids_raw, "FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS"),
            parse_ids(end_ids_raw, "FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS"),
        )
        self._argument_span_loss_cache_key = cache_key
        self._argument_span_loss_cache = settings
        return settings

    def _value_copy_loss_settings(self):
        weight_raw = os.environ.get("FASTDLLM_VALUE_COPY_LOSS_WEIGHT", "1.0")
        token_ids_raw = os.environ.get("FASTDLLM_VALUE_COPY_TOKEN_IDS", "")
        cache_key = (weight_raw, token_ids_raw)
        if getattr(self, "_value_copy_loss_cache_key", None) == cache_key:
            return self._value_copy_loss_cache

        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_VALUE_COPY_LOSS_WEIGHT={weight_raw!r}") from exc
        if weight <= 0:
            raise ValueError("FASTDLLM_VALUE_COPY_LOSS_WEIGHT must be positive")

        token_ids = []
        for item in token_ids_raw.replace(";", ",").replace(" ", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                token_ids.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid FASTDLLM_VALUE_COPY_TOKEN_IDS item={item!r}") from exc
        settings = (weight, tuple(sorted(set(token_ids))))
        self._value_copy_loss_cache_key = cache_key
        self._value_copy_loss_cache = settings
        return settings

    def _value_span_loss_settings(self):
        weight_raw = os.environ.get("FASTDLLM_VALUE_SPAN_LOSS_WEIGHT", "1.0")
        token_ids_raw = os.environ.get("FASTDLLM_VALUE_SPAN_TOKEN_IDS", "")
        cache_key = (weight_raw, token_ids_raw)
        if getattr(self, "_value_span_loss_cache_key", None) == cache_key:
            return self._value_span_loss_cache

        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_VALUE_SPAN_LOSS_WEIGHT={weight_raw!r}") from exc
        if weight <= 0:
            raise ValueError("FASTDLLM_VALUE_SPAN_LOSS_WEIGHT must be positive")

        token_ids = []
        for item in token_ids_raw.replace(";", ",").replace(" ", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                token_ids.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid FASTDLLM_VALUE_SPAN_TOKEN_IDS item={item!r}") from exc
        settings = (weight, tuple(sorted(set(token_ids))))
        self._value_span_loss_cache_key = cache_key
        self._value_span_loss_cache = settings
        return settings

    def _argument_span_mask_probability(self):
        prob_raw = os.environ.get("FASTDLLM_ARGUMENT_SPAN_MASK_PROB", "0.0")
        if getattr(self, "_argument_span_mask_probability_cache_key", None) == prob_raw:
            return self._argument_span_mask_probability_cache

        try:
            probability = float(prob_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_ARGUMENT_SPAN_MASK_PROB={prob_raw!r}") from exc
        if probability < 0 or probability > 1:
            raise ValueError("FASTDLLM_ARGUMENT_SPAN_MASK_PROB must be in [0, 1]")
        self._argument_span_mask_probability_cache_key = prob_raw
        self._argument_span_mask_probability_cache = probability
        return probability

    def _value_span_mask_probability(self):
        prob_raw = os.environ.get("FASTDLLM_VALUE_SPAN_MASK_PROB", "0.0")
        if getattr(self, "_value_span_mask_probability_cache_key", None) == prob_raw:
            return self._value_span_mask_probability_cache

        try:
            probability = float(prob_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid FASTDLLM_VALUE_SPAN_MASK_PROB={prob_raw!r}") from exc
        if probability < 0 or probability > 1:
            raise ValueError("FASTDLLM_VALUE_SPAN_MASK_PROB must be in [0, 1]")
        self._value_span_mask_probability_cache_key = prob_raw
        self._value_span_mask_probability_cache = probability
        return probability

    def _value_span_label_only_enabled(self):
        raw = os.environ.get("FASTDLLM_VALUE_SPAN_LABEL_ONLY", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _argument_span_active_mask(self, labels, require_ids=False):
        _, start_ids, end_ids = self._argument_span_loss_settings()
        if not start_ids or not end_ids:
            if require_ids:
                raise ValueError(
                    "Argument-span masking requires FASTDLLM_ARGUMENT_SPAN_START_TOKEN_IDS "
                    "and FASTDLLM_ARGUMENT_SPAN_END_TOKEN_IDS"
                )
            return None

        start = torch.isin(labels, torch.tensor(start_ids, device=labels.device, dtype=labels.dtype))
        end = torch.isin(labels, torch.tensor(end_ids, device=labels.device, dtype=labels.dtype))
        active_spans = torch.cumsum(start.to(torch.int32), dim=1) > torch.cumsum(end.to(torch.int32), dim=1)
        return active_spans & (labels != -100)

    def _value_span_active_mask(self, labels, require_ids=False):
        _, value_span_token_ids = self._value_span_loss_settings()
        if not value_span_token_ids:
            if require_ids:
                raise ValueError("Value-span labels require FASTDLLM_VALUE_SPAN_TOKEN_IDS")
            return None
        active_spans = self._argument_span_active_mask(labels, require_ids=require_ids)
        if active_spans is None:
            return None
        value_span_ids = torch.tensor(value_span_token_ids, device=labels.device, dtype=labels.dtype)
        return torch.isin(labels, value_span_ids) & active_spans

    def _argument_span_force_mask(self, labels):
        probability = self._argument_span_mask_probability()
        if probability == 0.0:
            return None

        active_spans = self._argument_span_active_mask(labels, require_ids=True)
        if active_spans is None:
            return None
        if probability == 1.0:
            return active_spans
        return active_spans & (torch.rand(labels.shape, device=labels.device) < probability)

    def _value_span_force_mask(self, labels):
        probability = self._value_span_mask_probability()
        if probability == 0.0:
            return None

        value_spans = self._value_span_active_mask(labels, require_ids=True)
        if value_spans is None:
            return None
        if probability == 1.0:
            return value_spans
        return value_spans & (torch.rand(labels.shape, device=labels.device) < probability)

    def _value_span_label_only_mask(self, labels):
        if not self._value_span_label_only_enabled():
            return None
        value_spans = self._value_span_active_mask(labels, require_ids=True)
        if value_spans is None:
            return None
        return value_spans & (labels != -100)

    def _argument_span_loss_weights(self, labels):
        weight, _, _ = self._argument_span_loss_settings()
        value_span_weight, value_span_token_ids = self._value_span_loss_settings()
        needs_argument_span = weight != 1.0
        needs_value_span = value_span_weight != 1.0 and bool(value_span_token_ids)
        if not needs_argument_span and not needs_value_span:
            return None

        active_spans = self._argument_span_active_mask(labels)
        if active_spans is None:
            return None
        loss_weights = torch.ones(labels.shape, device=labels.device, dtype=torch.float32)
        if needs_argument_span:
            loss_weights = torch.where(active_spans, torch.full_like(loss_weights, weight), loss_weights)
        if needs_value_span:
            value_span_ids = torch.tensor(value_span_token_ids, device=labels.device, dtype=labels.dtype)
            value_span = torch.isin(labels, value_span_ids) & active_spans
            value_span_weights = torch.full_like(loss_weights, value_span_weight)
            loss_weights = torch.where(value_span, torch.maximum(loss_weights, value_span_weights), loss_weights)
        else:
            value_span = torch.zeros_like(active_spans)

        debug_limit = int(os.environ.get("FASTDLLM_DEBUG_ARGUMENT_SPAN_LOSS", "0") or 0)
        debug_calls = getattr(self, "_debug_argument_span_loss_calls", 0)
        if debug_limit and debug_calls < debug_limit:
            _, start_ids, end_ids = self._argument_span_loss_settings()
            print(
                "[fastdllm-qwen35-debug] argument_span_loss "
                f"weight={weight} start_ids={len(start_ids)} end_ids={len(end_ids)} "
                f"value_span_weight={value_span_weight} value_span_ids={len(value_span_token_ids)} "
                f"valid={int((labels != -100).sum().detach().cpu())} "
                f"argument_span={int(active_spans.sum().detach().cpu())} "
                f"value_span={int(value_span.sum().detach().cpu())}",
                flush=True,
            )
            self._debug_argument_span_loss_calls = debug_calls + 1
        return loss_weights

    def _weighted_loss(self, logits, labels, vocab_size, loss_weights=None, **kwargs):
        structural_weight, structural_token_ids = self._structural_loss_settings()
        value_copy_weight, value_copy_token_ids = self._value_copy_loss_settings()
        if (
            loss_weights is None
            and (structural_weight == 1.0 or not structural_token_ids)
            and (value_copy_weight == 1.0 or not value_copy_token_ids)
        ):
            return self.loss_function(logits=logits, labels=labels, vocab_size=vocab_size, **kwargs)

        ignore_index = int(kwargs.get("ignore_index", -100))
        num_items_in_batch = kwargs.get("num_items_in_batch")
        logits = logits.float()
        shift_labels = kwargs.get("shift_labels")
        shift_loss_weights = None
        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
            if loss_weights is not None:
                loss_weights = nn.functional.pad(loss_weights, (0, 1), value=1.0)
                shift_loss_weights = loss_weights[..., 1:].contiguous()
        elif loss_weights is not None:
            shift_loss_weights = loss_weights

        flat_logits = logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1).to(flat_logits.device)
        per_token_loss = nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=ignore_index,
            reduction="none",
        )
        valid = flat_labels != ignore_index
        if shift_loss_weights is None:
            token_weights = torch.ones_like(per_token_loss)
        else:
            token_weights = shift_loss_weights.view(-1).to(flat_logits.device, dtype=per_token_loss.dtype)

        if structural_weight != 1.0 and structural_token_ids:
            structural_ids = torch.tensor(structural_token_ids, device=flat_labels.device, dtype=flat_labels.dtype)
            structural = torch.isin(flat_labels, structural_ids) & valid
            structural_weights = torch.full_like(token_weights, structural_weight)
            token_weights = torch.where(structural, torch.maximum(token_weights, structural_weights), token_weights)
        else:
            structural = torch.zeros_like(valid)

        if value_copy_weight != 1.0 and value_copy_token_ids:
            value_copy_ids = torch.tensor(value_copy_token_ids, device=flat_labels.device, dtype=flat_labels.dtype)
            value_copy = torch.isin(flat_labels, value_copy_ids) & valid
            value_copy_weights = torch.full_like(token_weights, value_copy_weight)
            token_weights = torch.where(value_copy, torch.maximum(token_weights, value_copy_weights), token_weights)
        else:
            value_copy = torch.zeros_like(valid)

        weighted_loss = per_token_loss * token_weights

        if num_items_in_batch is not None:
            loss = weighted_loss.sum()
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        else:
            denom = token_weights[valid].sum().clamp_min(1.0)
            loss = weighted_loss[valid].sum() / denom

        debug_limit = int(os.environ.get("FASTDLLM_DEBUG_STRUCTURAL_LOSS", "0") or 0)
        debug_calls = getattr(self, "_debug_structural_loss_calls", 0)
        if debug_limit and debug_calls < debug_limit:
            argument_weighted = (token_weights > 1.0) & valid
            print(
                "[fastdllm-qwen35-debug] structural_loss "
                f"weight={structural_weight} token_ids={len(structural_token_ids)} "
                f"valid={int(valid.sum().detach().cpu())} "
                f"structural={int(structural.sum().detach().cpu())} "
                f"value_copy={int(value_copy.sum().detach().cpu())} "
                f"weighted={int(argument_weighted.sum().detach().cpu())}",
                flush=True,
            )
            self._debug_structural_loss_calls = debug_calls + 1
        return loss

    def _flare_two_stream_enabled(self):
        return env_flag_enabled(FAST_DLLM_FLARE_TWO_STREAM_ENV, FLARE_TWO_STREAM_ENV)

    def _prepare_flare_doc_ids(self, input_ids, labels, attention_mask=None, doc_ids=None):
        if doc_ids is not None:
            doc_ids = doc_ids.to(device=input_ids.device, dtype=torch.long)
            if attention_mask is not None:
                doc_ids = torch.where(
                    attention_mask.to(device=input_ids.device).bool(),
                    doc_ids,
                    torch.full_like(doc_ids, -1),
                )
            return doc_ids
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid = attention_mask.to(device=input_ids.device).bool()
        row_doc_ids = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)[:, None]
        row_doc_ids = row_doc_ids.expand_as(input_ids)
        return torch.where(valid, row_doc_ids, torch.full_like(row_doc_ids, -1))

    def _build_flare_mask_views(
        self,
        labels,
        *,
        block_size,
        forced_argument_mask=None,
        forced_value_mask=None,
        provided_mask_indices=None,
    ):
        label_valid = labels != IGNORE_INDEX
        if provided_mask_indices is None:
            if labels.shape[1] % block_size:
                raise ValueError(
                    f"FLARE two-stream sequence length {labels.shape[1]} must be divisible by bd_size={block_size}"
                )
            batch_size, seq_len = labels.shape
            block_labels = labels.reshape(batch_size, seq_len // block_size, block_size)
            t = torch.rand(block_labels.shape[:2], device=labels.device)
            p_mask = ((1 - 1e-3) * t + 1e-3).unsqueeze(-1).expand_as(block_labels)
            mask_indices = torch.rand(block_labels.shape, device=labels.device) < p_mask
            mask_indices = mask_indices.reshape_as(labels)
        else:
            mask_indices = provided_mask_indices.to(device=labels.device, dtype=torch.bool)
            if mask_indices.shape != labels.shape:
                raise ValueError(
                    f"flare_mask_indices shape={tuple(mask_indices.shape)} does not match labels shape={tuple(labels.shape)}"
                )
        if forced_argument_mask is not None:
            mask_indices = mask_indices | forced_argument_mask.to(device=labels.device, dtype=torch.bool)
        if forced_value_mask is not None:
            mask_indices = mask_indices | forced_value_mask.to(device=labels.device, dtype=torch.bool)
        mask_view0 = mask_indices & label_valid
        mask_view1 = (~mask_indices) & label_valid
        return mask_view0, mask_view1

    def _compute_flare_losses(self, clean_logits, noisy_logits, labels, doc_ids, mask_view0, mask_view1):
        vocab_size = clean_logits.shape[-1]
        target_valid = (
            (doc_ids[:, :-1] >= 0)
            & (doc_ids[:, 1:] >= 0)
            & (doc_ids[:, :-1] == doc_ids[:, 1:])
            & (labels[:, 1:] != IGNORE_INDEX)
        )
        targets = labels[:, 1:].contiguous()
        ar_labels = torch.where(target_valid, targets, torch.full_like(targets, IGNORE_INDEX))
        ar_loss_sum = F.cross_entropy(
            clean_logits[:, :-1].contiguous().view(-1, vocab_size).float(),
            ar_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        ar_count = target_valid.sum().clamp_min(1)
        ar_loss = ar_loss_sum / ar_count

        batch_size = labels.shape[0]
        diff_mask0 = mask_view0[:, 1:] & target_valid
        diff_mask1 = mask_view1[:, 1:] & target_valid
        labels0 = torch.where(diff_mask0, targets, torch.full_like(targets, IGNORE_INDEX))
        labels1 = torch.where(diff_mask1, targets, torch.full_like(targets, IGNORE_INDEX))
        diff_loss0 = F.cross_entropy(
            noisy_logits[:batch_size, :-1].contiguous().view(-1, vocab_size).float(),
            labels0.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        diff_loss1 = F.cross_entropy(
            noisy_logits[batch_size:, :-1].contiguous().view(-1, vocab_size).float(),
            labels1.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        diff_loss = (diff_loss0 + diff_loss1) / ar_count
        return ar_loss + diff_loss, ar_loss, diff_loss, int(ar_count.detach().item())

    def _flare_two_stream_layer_forward(
        self,
        layer,
        clean_hidden,
        noisy_hidden,
        *,
        doc_ids,
        noisy_doc_ids,
        clean_mask,
        two_stream_mask,
        clean_position_ids,
        noisy_position_ids,
        block_size,
    ):
        clean_residual = clean_hidden
        noisy_residual = noisy_hidden
        clean_norm = _time_flare_section(
            "layernorm_clean",
            lambda: layer.input_layernorm(clean_hidden),
            clean_hidden,
        )
        noisy_norm = _time_flare_section(
            "layernorm_noisy",
            lambda: layer.input_layernorm(noisy_hidden),
            noisy_hidden,
        )

        if layer.layer_type == "linear_attention":
            gdn_route = flare_gdn_route()
            if gdn_route == "route_ii":
                clean_attn, noisy_attn = _time_flare_section(
                    "gdn_route_ii_clean_noisy",
                    lambda: clean_noisy_gdn_route_ii(
                        layer.linear_attn,
                        clean_norm,
                        noisy_norm,
                        noisy_doc_ids,
                        doc_ids,
                        block_size,
                        stride_blocks=flare_route_ii_stride_blocks(),
                    ),
                    clean_norm,
                    noisy_norm,
                )
            else:
                clean_attn, clean_boundary_states, clean_raw_qkv = _time_flare_section(
                    "gdn_clean_scan",
                    lambda: clean_gdn_docwise_with_boundaries(
                        layer.linear_attn,
                        clean_norm,
                        doc_ids,
                        block_size,
                    ),
                    clean_norm,
                )
                noisy_attn = _time_flare_section(
                    "gdn_noisy_scans",
                    lambda: noisy_gdn_route_i(
                        layer.linear_attn,
                        noisy_norm,
                        noisy_doc_ids,
                        doc_ids,
                        clean_boundary_states,
                        clean_raw_qkv,
                        block_size,
                    ),
                    noisy_norm,
                )
        else:
            clean_attn = _time_flare_section(
                "attention_clean",
                lambda: layer.self_attn(
                    hidden_states=clean_norm,
                    position_embeddings=self.model.rotary_emb(clean_norm, clean_position_ids),
                    attention_mask=clean_mask,
                    split_size=None,
                ),
                clean_norm,
            )
            combined_attn = _time_flare_section(
                "attention_noisy",
                lambda: layer.self_attn(
                    hidden_states=torch.cat([clean_norm.repeat(2, 1, 1), noisy_norm], dim=1),
                    position_embeddings=self.model.rotary_emb(noisy_norm, noisy_position_ids),
                    attention_mask=two_stream_mask,
                    split_size=clean_norm.shape[1],
                ),
                clean_norm,
                noisy_norm,
            )
            noisy_attn = combined_attn[:, clean_norm.shape[1] :]

        clean_hidden = clean_residual + clean_attn
        noisy_hidden = noisy_residual + noisy_attn
        clean_hidden = _time_flare_section(
            "mlp_clean",
            lambda: clean_hidden + layer.mlp(layer.post_attention_layernorm(clean_hidden)),
            clean_hidden,
        )
        noisy_hidden = _time_flare_section(
            "mlp_noisy",
            lambda: noisy_hidden + layer.mlp(layer.post_attention_layernorm(noisy_hidden)),
            noisy_hidden,
        )
        return clean_hidden, noisy_hidden

    def _flare_two_stream_training_forward(
        self,
        *,
        input_ids,
        labels,
        attention_mask=None,
        doc_ids=None,
        logits_to_keep=0,
        mask_id=None,
        flare_mask_indices=None,
    ):
        if _gdn_profile_enabled() or _flare_section_profile_enabled():
            _reset_gdn_profile()
            _reset_flare_section_profile()
            _cuda_synchronize_for_profile(input_ids)
            flare_forward_profile_start = time.time()
        else:
            flare_forward_profile_start = None
        mask_id = int(mask_id if mask_id is not None else self.config.mask_token_id)
        full_original_labels = labels.clone()
        value_label_only_mask = self._value_span_label_only_mask(full_original_labels)
        if value_label_only_mask is not None:
            labels = labels.clone()
            labels[~value_label_only_mask] = IGNORE_INDEX
        forced_argument_mask = self._argument_span_force_mask(full_original_labels)
        forced_value_mask = self._value_span_force_mask(full_original_labels)
        train_bd_size = self._resolve_train_bd_size(input_ids.shape[1], input_ids.device)
        self._set_active_train_bd_size(train_bd_size)
        block_size = int(self.model.bd_size)

        doc_ids = self._prepare_flare_doc_ids(input_ids, labels, attention_mask=attention_mask, doc_ids=doc_ids)
        mask_view0, mask_view1 = self._build_flare_mask_views(
            labels,
            block_size=block_size,
            forced_argument_mask=forced_argument_mask,
            forced_value_mask=forced_value_mask,
            provided_mask_indices=flare_mask_indices,
        )
        noisy_view0_ids = torch.where(mask_view0, torch.full_like(input_ids, mask_id), input_ids)
        noisy_view1_ids = torch.where(mask_view1, torch.full_like(input_ids, mask_id), input_ids)
        noisy_input_ids = torch.cat([noisy_view0_ids, noisy_view1_ids], dim=0)
        noisy_doc_ids = torch.cat([doc_ids, doc_ids], dim=0)

        clean_hidden = self.model.embed_tokens(input_ids)
        noisy_hidden = self.model.embed_tokens(noisy_input_ids)
        clean_mask = doc_causal_bool_mask(doc_ids)
        two_stream_mask = flare_two_stream_bool_mask(noisy_doc_ids, block_size)
        clean_position_ids = local_position_ids_from_doc_ids(doc_ids)
        noisy_position_ids = local_position_ids_from_doc_ids(noisy_doc_ids)

        if self.model.gradient_checkpointing and self.training:
            use_cache = False
        for layer in self.model.layers:
            if self.model.gradient_checkpointing and self.training:
                clean_hidden, noisy_hidden = _time_flare_section(
                    "layer_checkpoint_total",
                    lambda: self.model._gradient_checkpointing_func(
                        lambda clean, noisy, current_layer=layer: self._flare_two_stream_layer_forward(
                            current_layer,
                            clean,
                            noisy,
                            doc_ids=doc_ids,
                            noisy_doc_ids=noisy_doc_ids,
                            clean_mask=clean_mask,
                            two_stream_mask=two_stream_mask,
                            clean_position_ids=clean_position_ids,
                            noisy_position_ids=noisy_position_ids,
                            block_size=block_size,
                        ),
                        clean_hidden,
                        noisy_hidden,
                    ),
                    clean_hidden,
                    noisy_hidden,
                )
            else:
                clean_hidden, noisy_hidden = _time_flare_section(
                    "layer_total",
                    lambda: self._flare_two_stream_layer_forward(
                        layer,
                        clean_hidden,
                        noisy_hidden,
                        doc_ids=doc_ids,
                        noisy_doc_ids=noisy_doc_ids,
                        clean_mask=clean_mask,
                        two_stream_mask=two_stream_mask,
                        clean_position_ids=clean_position_ids,
                        noisy_position_ids=noisy_position_ids,
                        block_size=block_size,
                    ),
                    clean_hidden,
                    noisy_hidden,
                )

        clean_hidden = _time_flare_section(
            "final_norm_clean",
            lambda: self.model.norm(clean_hidden),
            clean_hidden,
        )
        noisy_hidden = _time_flare_section(
            "final_norm_noisy",
            lambda: self.model.norm(noisy_hidden),
            noisy_hidden,
        )
        clean_logits_full = _time_flare_section(
            "lm_head_clean_full",
            lambda: self.lm_head(clean_hidden),
            clean_hidden,
        )
        noisy_logits = _time_flare_section(
            "lm_head_noisy_full",
            lambda: self.lm_head(noisy_hidden),
            noisy_hidden,
        )
        loss, ar_loss, diff_loss, ar_count = _time_flare_section(
            "loss_compute",
            lambda: self._compute_flare_losses(
                clean_logits_full,
                noisy_logits,
                labels,
                doc_ids,
                mask_view0,
                mask_view1,
            ),
            clean_logits_full,
            noisy_logits,
        )
        self._last_flare_loss_parts = {
            "total": loss.detach(),
            "ar": ar_loss.detach(),
            "diff": diff_loss.detach(),
            "ar_count": ar_count,
            "mask_view0": int(mask_view0.sum().detach().cpu()),
            "mask_view1": int(mask_view1.sum().detach().cpu()),
        }

        debug_limit = int(os.environ.get("FASTDLLM_FLARE_DEBUG", "0") or 0)
        debug_calls = getattr(self, "_debug_flare_calls", 0)
        if debug_limit and debug_calls < debug_limit:
            print(
                "[fastdllm-qwen35-flare] "
                f"input_shape={tuple(input_ids.shape)} bd_size={block_size} "
                f"L_AR={float(ar_loss.detach().cpu()):.6g} "
                f"L_diff={float(diff_loss.detach().cpu()):.6g} "
                f"ar_count={ar_count} "
                f"mask_counts={int(mask_view0.sum().detach().cpu())}/{int(mask_view1.sum().detach().cpu())}",
                flush=True,
            )
            self._debug_flare_calls = debug_calls + 1

        if flare_forward_profile_start is not None:
            _cuda_synchronize_for_profile(clean_logits_full, noisy_logits)
            total_seconds = time.time() - flare_forward_profile_start
            gdn_profile = _snapshot_gdn_profile() or {}
            scan_seconds = float(gdn_profile.get("scan_seconds", 0.0))
            gdn_profile.update(
                {
                    "forward_total_seconds": total_seconds,
                    "forward_rest_seconds": total_seconds - scan_seconds,
                    "loss": float(loss.detach().float().cpu()),
                    "L_AR": float(ar_loss.detach().float().cpu()),
                    "L_diff": float(diff_loss.detach().float().cpu()),
                    "bd_size": int(block_size),
                }
            )
            print("[fastdllm-gdn-profile] " + json.dumps(gdn_profile, sort_keys=True), flush=True)
            section_profile = _snapshot_flare_section_profile()
            if section_profile is not None:
                section_profile.update(
                    {
                        "forward_total_seconds": total_seconds,
                        "loss": float(loss.detach().float().cpu()),
                        "L_AR": float(ar_loss.detach().float().cpu()),
                        "L_diff": float(diff_loss.detach().float().cpu()),
                        "bd_size": int(block_size),
                    }
                )
                print("[fastdllm-flare-section-profile] " + json.dumps(section_profile, sort_keys=True), flush=True)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        clean_logits = clean_logits_full[:, slice_indices, :]
        return Fast_dLLM_Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=clean_logits,
            past_key_values=None,
            hidden_states=clean_hidden,
            attentions=None,
            block_past_key_values=None,
        )

    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        if temperature <= 0:
            probs = torch.softmax(logits, dim=-1)
            return probs.argmax(dim=-1), probs

        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        probs = probs.masked_fill(indices_to_remove, 0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
        return sampled.reshape(probs.shape[:-1]), probs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        mask_id: Optional[int] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        flare_mask_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Fast_dLLM_Qwen3_5CausalLMOutputWithPast:
        if (
            self.training
            and labels is not None
            and input_ids is not None
            and inputs_embeds is None
            and self._flare_two_stream_enabled()
        ):
            return self._flare_two_stream_training_forward(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                doc_ids=doc_ids,
                logits_to_keep=logits_to_keep,
                mask_id=mask_id,
                flare_mask_indices=flare_mask_indices,
            )

        mdm_split_size = None
        loss_weights = None
        if self.training and labels is not None and input_ids is not None:
            debug_label_limit = int(os.environ.get("FASTDLLM_DEBUG_LABELS", "0") or 0)
            debug_label_calls = getattr(self, "_debug_label_calls", 0)
            should_debug_labels = debug_label_limit and debug_label_calls < debug_label_limit
            if should_debug_labels:
                valid_per_row = (labels != -100).sum(dim=1).detach().cpu().tolist()
                print(
                    "[fastdllm-qwen35-debug] pre_mdm "
                    f"input_shape={tuple(input_ids.shape)} "
                    f"valid_labels={valid_per_row}",
                    flush=True,
            )
            mask_id = int(mask_id if mask_id is not None else self.config.mask_token_id)
            full_original_labels = labels.clone()
            value_label_only_mask = self._value_span_label_only_mask(full_original_labels)
            if value_label_only_mask is not None:
                labels = labels.clone()
                labels[~value_label_only_mask] = -100
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()
            original_loss_weights = self._argument_span_loss_weights(full_original_labels)
            forced_argument_mask = self._argument_span_force_mask(full_original_labels)
            forced_value_mask = self._value_span_force_mask(full_original_labels)
            train_bd_size = self._resolve_train_bd_size(input_ids.shape[1], input_ids.device)
            self._set_active_train_bd_size(train_bd_size)
            noisy_input_ids = input_ids.clone()
            block_input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            bsz_blocks, block_len = block_input_ids.shape
            t = torch.rand((bsz_blocks,), device=input_ids.device)
            p_mask = ((1 - 1e-3) * t + 1e-3)[:, None].repeat(1, block_len)
            mask_indices = torch.rand((bsz_blocks, block_len), device=input_ids.device) < p_mask
            if forced_argument_mask is not None:
                mask_indices = mask_indices | forced_argument_mask.reshape(bsz_blocks, block_len)
            if forced_value_mask is not None:
                mask_indices = mask_indices | forced_value_mask.reshape(bsz_blocks, block_len)
            x_t = torch.where(mask_indices, mask_id, block_input_ids).reshape(labels.shape)
            noisy_input_ids[labels != -100] = x_t[labels != -100]
            masked_labels = labels.clone()
            masked_labels[noisy_input_ids != mask_id] = -100
            if original_loss_weights is not None:
                masked_loss_weights = original_loss_weights.clone()
                masked_loss_weights[masked_labels == -100] = 1.0
            input_ids_main = torch.cat([noisy_input_ids, block_input_ids.reshape(labels.shape)], dim=1)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()
            complementary_block_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            complementary_x_t = torch.where(~mask_indices, mask_id, complementary_block_ids).reshape(original_labels.shape)
            complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
            complementary_labels[complementary_noisy_input_ids != mask_id] = -100
            if original_loss_weights is not None:
                complementary_loss_weights = original_loss_weights.clone()
                complementary_loss_weights[complementary_labels == -100] = 1.0
            complementary_input_ids = torch.cat(
                [complementary_noisy_input_ids, complementary_block_ids.reshape(complementary_labels.shape)], dim=1
            )
            input_ids = torch.cat([input_ids_main, complementary_input_ids], dim=0)
            labels = torch.cat([masked_labels, complementary_labels], dim=0)
            if original_loss_weights is not None:
                loss_weights = torch.cat([masked_loss_weights, complementary_loss_weights], dim=0)
            mdm_split_size = labels.shape[1]
            if should_debug_labels:
                valid_per_row = (labels != -100).sum(dim=1).detach().cpu().tolist()
                print(
                    "[fastdllm-qwen35-debug] post_mdm "
                    f"input_shape={tuple(input_ids.shape)} "
                    f"valid_labels={valid_per_row} "
                    f"bd_size={self.model.bd_size} "
                    f"value_label_only={int(value_label_only_mask.sum().detach().cpu()) if value_label_only_mask is not None else 0} "
                    f"forced_argument_mask={int(forced_argument_mask.sum().detach().cpu()) if forced_argument_mask is not None else 0} "
                    f"forced_value_mask={int(forced_value_mask.sum().detach().cpu()) if forced_value_mask is not None else 0}",
                    flush=True,
                )
                self._debug_label_calls = debug_label_calls + 1

        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mdm_split_size=mdm_split_size,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if self.training and mdm_split_size is not None:
            hidden_states = hidden_states[:, :mdm_split_size, :]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            if loss_weights is None:
                loss_weights = self._argument_span_loss_weights(labels)
            loss = self._weighted_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                loss_weights=loss_weights,
                **kwargs,
            )
        return Fast_dLLM_Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=None,
            block_past_key_values=outputs.block_past_key_values,
        )
