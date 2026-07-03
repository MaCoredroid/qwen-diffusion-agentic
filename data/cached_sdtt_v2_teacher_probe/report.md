# Cached-SDTT V2 Teacher Corpus

Teacher: final v2 adapter. Source rollouts: diffusion careful + live Qwen-native grammar. Kept turns are audit-clean exact-args only.

| Metric | Value |
|---|---:|
| Episodes rolled out | 130 |
| Turns seen | 269 |
| Exact/audit-clean turns | 223 |
| Cached records | 160 |
| Cached target tokens | 1146 |
| Rejected targetless exact turns | 63 |

Configuration: `teacher_steps=2`, `top_k=256`, teacher-forced leftmost gold value-token commits per span, sparse top-k reverse-KL caveat.

Git HEAD: `74e0c145d5aedf899c7f9b7ee3b008d55a0e4928`
Output JSONL: `data/cached_sdtt_v2_teacher_probe/cached_sdtt_records.jsonl`
