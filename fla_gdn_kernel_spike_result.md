# FLA fused GDN kernel spike result (2026-06-30)

Scope: Step 0 gate from `fla_kernel_feasibility.md` only. No model integration or promotion was done.

## Environment
- Python env: `.venv-fastdllm`
- `flash-linear-attention`: `0.5.1` (bare install; Torch/Triton remained unchanged)
- `torch`: `2.12.1+cu130`
- `triton`: `3.7.1`
- GPU: NVIDIA GeForce RTX 5090, capability `sm_120`

## Script
- `scripts/spike_fla_gdn_kernel.py`
- Direct FLA call: `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
- Shape: one full training GDN block, `[B=1, T=1024, HV=32, K=128, V=128]`
- Dtype: bf16 q/k/v/beta/initial_state, float32 raw per-token `g`
- Initial state: non-zero and `requires_grad=True`
- Flags:
  - `use_qk_l2norm_in_kernel=False`
  - `use_beta_sigmoid_in_kernel=False`
  - `scale=1/sqrt(d_k)=0.08838834764831843`
  - `allow_neg_eigval=False`
  - no pre-cumsum of `g`; FLA owns the per-chunk cumsum

Command:
```bash
.venv-fastdllm/bin/python scripts/spike_fla_gdn_kernel.py --seq-len 1024 --print-json
```

## Checks
- #607 / Blackwell crash: PASS. No `tmem_store` crash and no `no kernel image` error on fwd+bwd.
- #734 cumsum crash: PASS. FLA internal `chunk_local_cumsum` completed.
- Finite grads: PASS for `dq`, `dk`, `dv`, `dbeta`, `dg`, and `dh0`.
- Parity vs local `_torch_chunk_gated_delta_rule_impl`: PASS at `rtol=1e-2`, `atol=2e-2`.

Selected full-block parity maxima:
| tensor | max abs | allclose |
| --- | ---: | --- |
| output | 0.000244140625 | true |
| final_state | 0.0010833479464054108 | true |
| dq | 0.0000152587890625 | true |
| dk | 0.00048828125 | true |
| dv | 0.000244140625 | true |
| dbeta | 0.000244140625 | true |
| dg | 0.00045932456851005554 | true |
| dh0 | 0.00000762939453125 | true |

## Verdict
GREEN: parity holds and the FLA fwd+bwd path does not crash on `sm_120`. Ready for a separate integration step, pending monitor red-team.

## Status: PARKED (2026-06-30, user directive)
Monitor red-team: GREEN confirmed, no holes (#607 retired on triton 3.7.1; dh0 + all grads finite; parity within bf16 tol with margin). Caveat: this proves *kernel correctness* on one block/config, not end-to-end training parity or the util win — those belong to the (deferred) integration + validation gate. **Integration is PARKED, not abandoned** — it is ~0.5–1.5 days and only pays off if we run more training; resume per the STATUS block in `fla_kernel_feasibility.md` when further training is on the table.
