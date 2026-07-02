# Repaired Multi-Turn SFT Warm-Start

Self-generated rows are rewritten as one compact user prompt plus one assistant target.
Prior assistant calls are retained only as user-visible context text, so the trainer labels only the final audited assistant target.

## Counts

- Final rows: `98`
- Repaired self-generated rows: `49`
- Retention rows copied unchanged: `49`
- Source counts: `{'google-research-datasets/mbpp:full:train': 24, 'openai/gsm8k:main:train': 25, 'selfgen_multiturn_exact_audited': 49}`

## Token Audit

- Original over 512 tokens: `49`
- Repaired over 512 tokens: `0`
- Original partial labels at 512-left: `4`
- Repaired partial labels at 512-left: `0`
- Repaired rows with full labels kept at 512-left: `98/98`
