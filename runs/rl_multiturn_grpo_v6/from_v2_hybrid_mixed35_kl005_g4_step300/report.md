# Multi-Turn diffu-GRPO Pilot

- Warm start: `/home/mark/qwen_diffusion/runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model`
- Steps: `113`
- Group size: `4`
- Grouping: `mixed adjacent public episodes`
- Nonzero-advantage steps: `111/113`
- Policy replay tokens: `7246`
- Policy value tokens: `7246`
- Policy free tokens: `0`
- Grammar-forced tokens masked from policy loss: `31670`
- KL-to-base coefficient: `0.05`
- KL early stop: last `50` mean > `0.05`
- Retention probe cadence: every `50` steps, limit `10`
- Early stopped: `True` (kl_last_window_mean)
- Mean step reward: `0.9108`
- Output adapter: `/home/mark/qwen_diffusion/runs/rl_multiturn_grpo_v6/from_v2_hybrid_mixed35_kl005_g4_step300/adapter_model`

Rollouts used `hybrid_clean` decode with audited ToolRL-style reward. The update is a raw full-vocab logprob replay approximation over generated parameter-value/free tokens only; grammar-forced structure is excluded from the policy loss.
