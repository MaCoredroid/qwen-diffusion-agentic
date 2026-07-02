# S1 VRAM Smoke Report

Ceiling: `30.5 GiB = 31232 MiB`.

| config | peak allocated MiB | peak reserved MiB | runtime s | verdict |
| --- | ---: | ---: | ---: | --- |
| r64_alpha128 | 19838.1 | 20576.0 | 17.44 | PASS |
| r128_alpha256 | 21081.8 | 21630.0 | 18.34 | PASS |

Result: r=64 S1 launch is cleared; r=128/alpha=256 is banked for one-shot escalation if S1 gates require it.
