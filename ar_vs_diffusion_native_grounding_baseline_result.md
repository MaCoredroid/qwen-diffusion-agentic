# AR vs diffusion native grounding baseline (2026-06-30)

Scope: controlled diagnostic only. No promotion and no FLA integration.

## Setup
- Model weights: converted base B@1000 adapter `runs/flare_stage1_ab_pilot/two_stream_B_s1024_step1000` on `models/qwen3.5-9b-fastdllm-init`.
- Cases: Qwen-native tool-call eval under `data/toolcall_eval_native/`.
- Label-free: gold stripped before generation; gold used only for scoring.
- AR raw: new `scripts/eval_fastdllm_ar_toolcall_cases.py`, causal full-context greedy forward, `use_cache=False`, mask token banned.
- AR + decoder: same AR loop plus the same Qwen-native prefix grammar (`live_tool_json_top_token` / schema-only prefix legality) with forced `<tool_call>\n`, `topk=1024`.
- Diffusion + decoder: existing B@1000 live grammar result plus a new one-call-24 extension for the full-pool table.
- 27B teacher: skipped; `http://127.0.0.1:30000` was not serving and no SGLang/Qwen3.6 process was running.

## Results
Cells are `valid / exact-seq / exact-args`.

### Comparable 28-subset
| Mode | onecall 8 | multicall 12 | teacher 8 | total 28 |
| --- | ---: | ---: | ---: | ---: |
| Diffusion + live decoder | 8 / 7 / 5 | 12 / 11 / 10 | 8 / 7 / 4 | 28 / 25 / 19 |
| AR raw | 7 / 6 / 5 | 11 / 10 / 9 | 7 / 6 / 4 | 25 / 22 / 18 |
| AR + native grammar | 8 / 7 / 5 | 12 / 11 / 10 | 8 / 7 / 4 | 28 / 25 / 19 |
| 27B teacher | skipped | skipped | skipped | skipped |

### Full pool
| Mode | onecall 24 | multicall 12 | teacher 8 | total 44 |
| --- | ---: | ---: | ---: | ---: |
| Diffusion + live decoder | 24 / 23 / 19 | 12 / 11 / 10 | 8 / 7 / 4 | 44 / 41 / 33 |
| AR raw | 21 / 19 / 16 | 11 / 10 / 9 | 7 / 6 / 4 | 39 / 35 / 29 |
| AR + native grammar | 24 / 23 / 19 | 12 / 11 / 10 | 8 / 7 / 4 | 44 / 41 / 33 |
| 27B teacher | skipped | skipped | skipped | skipped |

AR+grammar had zero unsafe grammar fallbacks on all slices:
- onecall 24: active 2613, replacements 1, unsafe 0
- multicall 12: active 2030, replacements 3, unsafe 0
- teacher 8: active 924, replacements 1, unsafe 0

## Per-row exact-args
`D` = diffusion + live decoder, `R` = AR raw, `G` = AR + native grammar. `1` means exact args.

| Pool | Slice | Row | ID | D | R | G |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| full | onecall | 1 | 14657d01-d6d1-46df-8eb1-7267ba820683 | 1 | 0 | 1 |
| full | onecall | 2 | 7d99abac-f27f-4ed2-a9ea-31faab5ad5e8 | 0 | 0 | 0 |
| full | onecall | 3 | 8e0d5c89-d6d2-4724-9e5c-249860fc3cfc | 1 | 1 | 1 |
| full | onecall | 4 | edfc63f9-9534-4205-87ca-e37f3630eabe | 1 | 1 | 1 |
| full | onecall | 5 | 610f6af1-e515-4167-8c16-930b22025e9a | 0 | 0 | 0 |
| full | onecall | 6 | 5ba3ba10-f1d7-4911-8a8d-42b947c1ff32 | 1 | 1 | 1 |
| full | onecall | 7 | 4195cbf2-2257-4963-9fa0-1a5e55ea4a35 | 0 | 1 | 0 |
| full | onecall | 8 | 64f1174b-dca3-4863-9202-5888503701e1 | 1 | 1 | 1 |
| full | onecall | 9 | 2b4b37c2-13a3-4d20-be44-1302da398728 | 1 | 1 | 1 |
| full | onecall | 10 | 1e9a5790-61b9-485f-bc6e-6d9d7335bb0d | 1 | 1 | 1 |
| full | onecall | 11 | 84bff146-4486-42c2-98da-a7b22919ce2d | 0 | 0 | 0 |
| full | onecall | 12 | b2165b46-1320-48de-a1f1-3202f99168d6 | 1 | 1 | 1 |
| full | onecall | 13 | cb260921-b346-4076-9232-a175cf82d32e | 0 | 0 | 0 |
| full | onecall | 14 | 24b4de96-700f-42e1-b42f-d7f3fb7c563b | 1 | 1 | 1 |
| full | onecall | 15 | d1d0816c-0eab-4eca-ba22-a5ec8dff43fe | 1 | 0 | 1 |
| full | onecall | 16 | 33c3f0d3-00eb-4b77-b86c-0383f7747ad8 | 1 | 1 | 1 |
| full | onecall | 17 | 3673692e-4ef3-4b4a-b3cc-e9d4a58607f7 | 1 | 0 | 1 |
| full | onecall | 18 | d31cc0b1-5a6d-43a5-950b-a4649096103a | 1 | 1 | 1 |
| full | onecall | 19 | 7293a34f-a794-4963-8a9b-5d43da04b37a | 1 | 1 | 1 |
| full | onecall | 20 | c82abc25-206c-4776-a1b1-d6fbc5769bce | 1 | 1 | 1 |
| full | onecall | 21 | faba3c70-215b-4567-a5aa-d3e4c53674c2 | 1 | 1 | 1 |
| full | onecall | 22 | 58359f63-4f59-4a09-8abb-7a4ee3c08cce | 1 | 1 | 1 |
| full | onecall | 23 | 85bc3350-6744-4699-9cd3-d552d612c677 | 1 | 1 | 1 |
| full | onecall | 24 | 5df07ac2-1a79-43f5-9a7f-93b97ef4f020 | 1 | 0 | 1 |
| full | multicall | 1 | 85f6c398-69c7-4df2-aed1-29d614a93a26 | 0 | 0 | 0 |
| full | multicall | 2 | 89ef3c87-66bd-46ee-9297-15398fd9a235 | 1 | 1 | 1 |
| full | multicall | 3 | c483f963-8a29-4ff0-a684-89be0d0f2843 | 0 | 0 | 0 |
| full | multicall | 4 | 81ad724a-bb74-420f-8221-91557b7e5930 | 1 | 1 | 1 |
| full | multicall | 5 | d1bff923-dbd1-45da-b0a0-7bd33c9cbc46 | 1 | 1 | 1 |
| full | multicall | 6 | d43227ba-5022-4cfc-8b70-24fae64d82dd | 1 | 1 | 1 |
| full | multicall | 7 | 3f440c20-b332-48e2-aaa5-a7bfb0781ae9 | 1 | 1 | 1 |
| full | multicall | 8 | e279e98f-095a-4d44-9c2d-170b3cfdc4bb | 1 | 1 | 1 |
| full | multicall | 9 | ec0e73f1-1f85-4963-b840-4a7e76b1c5b3 | 1 | 1 | 1 |
| full | multicall | 10 | 5790a757-bbe7-49d7-9fbd-d98cf3e0fd45 | 1 | 1 | 1 |
| full | multicall | 11 | 6de2be31-985e-413a-ae33-4c1140070920 | 1 | 1 | 1 |
| full | multicall | 12 | adc48a37-6341-4ea6-972a-8ec2b5421321 | 1 | 0 | 1 |
| full | teacher | 1 | 7d99abac-f27f-4ed2-a9ea-31faab5ad5e8 | 0 | 0 | 0 |
| full | teacher | 2 | 610f6af1-e515-4167-8c16-930b22025e9a | 0 | 0 | 0 |
| full | teacher | 3 | 4195cbf2-2257-4963-9fa0-1a5e55ea4a35 | 0 | 1 | 0 |
| full | teacher | 4 | 84bff146-4486-42c2-98da-a7b22919ce2d | 0 | 0 | 0 |
| full | teacher | 5 | 3673692e-4ef3-4b4a-b3cc-e9d4a58607f7 | 1 | 0 | 1 |
| full | teacher | 6 | d31cc0b1-5a6d-43a5-950b-a4649096103a | 1 | 1 | 1 |
| full | teacher | 7 | 7293a34f-a794-4963-8a9b-5d43da04b37a | 1 | 1 | 1 |
| full | teacher | 8 | c82abc25-206c-4776-a1b1-d6fbc5769bce | 1 | 1 | 1 |

## Interpretation
On the exact same weights and native cases, live constrained decoding removes essentially the remaining
serialization gap: AR+grammar and diffusion+grammar are identical at 19/28 and 33/44. AR raw is already strong
but loses structural validity/sequence on a few rows (18/28, 29/44). The current 19/28 SOTA is therefore not a
large diffusion-vs-AR grounding deficit under the live decoder; it is the shared value-content ceiling of this
B@1000 adapter plus schema-only grammar.
