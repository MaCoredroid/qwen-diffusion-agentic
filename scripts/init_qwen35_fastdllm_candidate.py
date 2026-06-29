#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import time
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path("/home/mark/qwen_diffusion")
DEFAULT_OUT = ROOT / "models/qwen3.5-9b-fastdllm-init"
MASK_TOKEN = "|<MASK>|"


CONFIGURATION_PY = '''"""Fast_dLLM Qwen3.5 text-only configuration scaffold."""

from transformers.configuration_utils import PretrainedConfig


class Fast_dLLM_Qwen3_5Config(PretrainedConfig):
    model_type = "Fast_dLLM_Qwen3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise",
        "layers.*.linear_attn.in_proj_z": "colwise",
        "layers.*.linear_attn.out_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        rope_parameters=None,
        bd_size=32,
        mask_token="|<MASK>|",
        mask_token_id=None,
        gdn_mode="option_a_causal_gdn",
        diffusion_bridge_status="scaffold",
        source_model=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.layer_types = layer_types or [
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(num_hidden_layers)
        ]
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        }
        self.bd_size = bd_size
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id
        self.gdn_mode = gdn_mode
        self.diffusion_bridge_status = diffusion_bridge_status
        self.source_model = source_model

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
'''


MODELING_PY = '''"""Fast_dLLM Qwen3.5 text-only modeling scaffold.

This file is intentionally a guardrail, not a complete model. It lets
AutoConfig load the converted candidate under the Fast-DLLM training
environment while preventing accidental training before the Qwen3.5 GDN bridge
is implemented.
"""

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from .configuration import Fast_dLLM_Qwen3_5Config


FAST_DLLM_QWEN3_5_BRIDGE_STATUS = "scaffold"
FAST_DLLM_QWEN3_5_GDN_MODE = "option_a_causal_gdn"


class Fast_dLLM_Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = Fast_dLLM_Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]


class Fast_dLLM_Qwen3_5ForCausalLM(Fast_dLLM_Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Fast_dLLM_Qwen3_5Config):
        raise NotImplementedError(
            "Fast_dLLM_Qwen3_5ForCausalLM is a scaffold. Implement the Qwen3.5 "
            "GDN/full-attention diffusion bridge before launching training. "
            "Option A should keep GDN causal and apply block-diffusion masks to "
            "full-attention layers only."
        )


class Fast_dLLM_Qwen3_5Model(Fast_dLLM_Qwen3_5PreTrainedModel):
    def __init__(self, config: Fast_dLLM_Qwen3_5Config):
        raise NotImplementedError(
            "Fast_dLLM_Qwen3_5Model is not implemented yet; this candidate is "
            "configuration/tokenizer scaffolding only."
        )
'''


def to_dict_config(config):
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def load_raw_text_config(model, local_files_only):
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
    raw = to_dict_config(cfg)
    text = raw.get("text_config") or raw
    if hasattr(cfg, "text_config"):
        text = to_dict_config(cfg.text_config)
    return raw, text


def build_fast_config(raw_config, text_config, tokenizer, args):
    layer_types = text_config.get("layer_types") or [
        "linear_attention" if bool((i + 1) % 4) else "full_attention"
        for i in range(text_config.get("num_hidden_layers", 32))
    ]
    return {
        "architectures": ["Fast_dLLM_Qwen3_5ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration.Fast_dLLM_Qwen3_5Config",
            "AutoModel": "modeling.Fast_dLLM_Qwen3_5Model",
            "AutoModelForCausalLM": "modeling.Fast_dLLM_Qwen3_5ForCausalLM",
        },
        "model_type": "Fast_dLLM_Qwen3_5",
        "source_model": args.raw_model,
        "source_architectures": raw_config.get("architectures"),
        "diffusion_bridge_status": "scaffold",
        "gdn_mode": "option_a_causal_gdn",
        "bd_size": args.bd_size,
        "mask_token": args.mask_token,
        "mask_token_id": tokenizer.get_vocab().get(args.mask_token),
        "vocab_size": text_config.get("vocab_size", 248320),
        "hidden_size": text_config.get("hidden_size", 4096),
        "intermediate_size": text_config.get("intermediate_size", 12288),
        "num_hidden_layers": text_config.get("num_hidden_layers", 32),
        "num_attention_heads": text_config.get("num_attention_heads", 16),
        "num_key_value_heads": text_config.get("num_key_value_heads", 4),
        "hidden_act": text_config.get("hidden_act", "silu"),
        "max_position_embeddings": text_config.get("max_position_embeddings", 262144),
        "initializer_range": text_config.get("initializer_range", 0.02),
        "rms_norm_eps": text_config.get("rms_norm_eps", 1e-6),
        "use_cache": text_config.get("use_cache", True),
        "tie_word_embeddings": raw_config.get("tie_word_embeddings", text_config.get("tie_word_embeddings", False)),
        "attention_bias": text_config.get("attention_bias", False),
        "attention_dropout": text_config.get("attention_dropout", 0.0),
        "head_dim": text_config.get("head_dim", 256),
        "linear_conv_kernel_dim": text_config.get("linear_conv_kernel_dim", 4),
        "linear_key_head_dim": text_config.get("linear_key_head_dim", 128),
        "linear_value_head_dim": text_config.get("linear_value_head_dim", 128),
        "linear_num_key_heads": text_config.get("linear_num_key_heads", 16),
        "linear_num_value_heads": text_config.get("linear_num_value_heads", 32),
        "layer_types": layer_types,
        "rope_parameters": text_config.get("rope_parameters"),
        "bos_token_id": text_config.get("bos_token_id"),
        "eos_token_id": text_config.get("eos_token_id"),
        "pad_token_id": text_config.get("pad_token_id") or text_config.get("eos_token_id"),
        "torch_dtype": text_config.get("dtype") or text_config.get("torch_dtype") or "bfloat16",
        "transformers_version": "4.53.1",
    }


def detect_bridge_metadata(out_dir):
    modeling = out_dir / "modeling.py"
    if not modeling.exists():
        return "scaffold", "option_a_causal_gdn"
    text = modeling.read_text(encoding="utf-8", errors="ignore")
    status_match = re.search(r"FAST_DLLM_QWEN3_5_BRIDGE_STATUS\s*=\s*['\"]([^'\"]+)['\"]", text)
    gdn_match = re.search(r"FAST_DLLM_QWEN3_5_GDN_MODE\s*=\s*['\"]([^'\"]+)['\"]", text)
    status = status_match.group(1) if status_match else "unknown"
    gdn_mode = gdn_match.group(1) if gdn_match else "unknown"
    return status, gdn_mode


def normalize_tokenizer_config(out_dir, mask_token):
    path = out_dir / "tokenizer_config.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    extra = data.pop("extra_special_tokens", None)
    additional = data.get("additional_special_tokens") or []
    if not isinstance(additional, list):
        additional = []
    if isinstance(extra, list):
        for token in extra:
            if token not in additional:
                additional.append(token)
    if mask_token not in additional:
        additional.append(mask_token)

    data["additional_special_tokens"] = additional
    data["transformers_version"] = "4.53.1"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bd-size", type=int, default=32)
    parser.add_argument("--mask-token", default=MASK_TOKEN)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    local_files_only = args.local_files_only and not args.allow_download
    out_dir = args.out_dir
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_config, text_config = load_raw_text_config(args.raw_model, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.raw_model, trust_remote_code=True, local_files_only=local_files_only)
    added = tokenizer.add_special_tokens({"additional_special_tokens": [args.mask_token]})
    tokenizer.save_pretrained(out_dir)
    normalize_tokenizer_config(out_dir, args.mask_token)

    bridge_status, gdn_mode = detect_bridge_metadata(out_dir)
    fast_config = build_fast_config(raw_config, text_config, tokenizer, args)
    fast_config["diffusion_bridge_status"] = bridge_status
    fast_config["gdn_mode"] = gdn_mode
    (out_dir / "config.json").write_text(json.dumps(fast_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    configuration_path = out_dir / "configuration.py"
    modeling_path = out_dir / "modeling.py"
    if not configuration_path.exists():
        configuration_path.write_text(CONFIGURATION_PY, encoding="utf-8")
    if not modeling_path.exists():
        modeling_path.write_text(MODELING_PY, encoding="utf-8")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_model": args.raw_model,
        "out_dir": str(out_dir),
        "bd_size": args.bd_size,
        "mask_token": args.mask_token,
        "mask_token_id": fast_config["mask_token_id"],
        "tokenizer_added_count": added,
        "bridge_status": bridge_status,
        "gdn_mode": gdn_mode,
        "has_weights": bool(list(out_dir.glob("*.safetensors")) or list(out_dir.glob("*.bin"))),
        "note": (
            "Existing implemented bridge metadata preserved."
            if bridge_status == "implemented"
            else "Tokenizer/config scaffold only. Model weights and Qwen3.5 GDN Fast-DLLM modeling are not implemented."
        ),
    }
    (out_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
