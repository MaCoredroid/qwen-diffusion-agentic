"""Fast_dLLM Qwen3.5 text-only configuration scaffold."""

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
