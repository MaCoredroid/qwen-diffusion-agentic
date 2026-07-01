# FLARE graded eval report

## Scope

- Do not promote.
- Quantization: matched bf16, no 4-bit.
- Memory fix path: lossless pre-repeat K/V cache plus allocator `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Telecom diffusion run was intentionally stopped after 20 rows because strict binary was 0-vs-0 and the speed gap was already clear.

## Cache Guard

- Real 9B T1 cache ON/OFF argmax guard: pass, 0 flips.
- T3 generation exactness: token exact and byte exact.
- Real guard peak reserved memory: 17.305 GiB.

## Clean Diffusion Memory

- Telecom memfix run peak at max prompt 8336 tokens: 17.557 GiB CUDA reserved, 17.469 GiB allocated.
- Live nvidia-smi during run: about 24.9 GiB used including the 6.3 GiB desktop baseline, so the diffusion process added about 18.6 GiB.
- After stopping the run: GPU returned to 6.3 GiB desktop baseline.

## Telecom Partial-Credit Signal

Strict binary remains uninformative:

- AR: 0/36 binary; partial action 4/84 = 0.048.
- Diffusion stopped sample: 0/20 binary; partial action 0/44 = 0.000.

Speed:

- AR backend speed: 89.162 tok/s.
- Diffusion backend speed: 2.255 tok/s.
- Diffusion / AR speed ratio: 0.025x, about 39.5x slower than AR.
- Diffusion denoise-forwards/token: 0.966.

Diffusion failures in the stopped sample:

- stop_turn_budget: 20
- invalid_call: 19
- wrong_tool: 20
- tool_execution_error: 16

## Easier Domain Signal

tau2 mock gives a nonzero AR regime:

- AR: 6/6 binary, partial action 6/6 = 1.000.
- Diffusion memfix: 1/6 binary, partial action 2/6 = 0.333.
- Diffusion constrained lane: 1/2 binary, partial action 1/2 = 0.500.

Speed:

- AR backend speed: 91.808 tok/s.
- Diffusion backend speed: 8.765 tok/s.
- Diffusion / AR speed ratio: 0.095x, about 10.5x slower than AR.
- Diffusion denoise-forwards/token: 0.989.
- Diffusion peak at max prompt 1006 tokens: 16.971 GiB CUDA reserved.

## Artifacts

- Telecom comparison JSON: `runs/agentic_eval/tau2_real_solo_memfix_comparison.json`
- Telecom comparison report: `runs/agentic_eval/tau2_real_solo_memfix_report.md`
- Easier-domain comparison JSON: `runs/agentic_eval/tau2_mock_easier_domain_memfix_comparison.json`
- Easier-domain comparison report: `runs/agentic_eval/tau2_mock_easier_domain_memfix_report.md`
- Diffusion mock memfix JSONL: `runs/agentic_eval/tau2_mock_diffusion9b_memfix.jsonl`
- Diffusion mock memfix summary: `runs/agentic_eval/tau2_mock_diffusion9b_memfix.summary.json`
