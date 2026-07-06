# W2 N=50 stock-AR vs diffusion — resolve@1 (paired)

pool_sha256 `fe1973937dfb500b…`  N=50  scoring: AR=ok diffusion=ok

## PRIMARY — paired resolve@1 McNemar
- AR resolved: **19/50**   diffusion resolved: **2/50**
- both=2  AR-only(b)=17  diffusion-only(c)=0  neither=31
- net (diffusion−AR) = **-17**   McNemar exact 2-sided p = **0.0000**
- **PARITY CLAIM (|net|<=2 AND p>=0.05): NO**
  - AR-only ids: ['django__django-12713', 'django__django-13315', 'django__django-13933', 'django__django-14089', 'django__django-14855', 'django__django-15104', 'django__django-15561', 'django__django-16082', 'django__django-16429', 'django__django-16493', 'matplotlib__matplotlib-13989', 'matplotlib__matplotlib-24970', 'pydata__xarray-4075', 'scikit-learn__scikit-learn-12585', 'scikit-learn__scikit-learn-14053', 'sympy__sympy-20154', 'sympy__sympy-23824']

## SECONDARY
### Throughput (episodes / GPU-h at run concurrency)
- ar: **99.61 eps/GPU-h**  (n=50, wall=1807s, c=4)
- diffusion: **21.41 eps/GPU-h**  (n=50, wall=8407s, c=4)
### Tokens + loop-halt covariates
| arm | eps | resolved | input_tok | output_tok | loop_halts | budget | timeouts | empty | med_turns |
|---|---|---|---|---|---|---|---|---|---|
| ar | 50 | 19 | 16,443,224 | 133,734 | 0 | 0 | 0 | 4 | 31 |
| diffusion | 50 | 2 | 26,729,835 | 115,351 | 26 | 1 | 0 | 35 | 25 |
### Loop-halt resolve split (covariate, pre-registered)
- ar: post-resolve halts=0  pre-resolve halts=0
- diffusion: post-resolve halts=1  pre-resolve halts=25
### Per-repo resolved (arm: resolved/total)
| repo | AR | diffusion |
|---|---|---|
| astropy/astropy | 0/2 | 0/2 |
| django/django | 11/22 | 1/22 |
| matplotlib/matplotlib | 2/3 | 0/3 |
| mwaskom/seaborn | 0/1 | 0/1 |
| psf/requests | 0/1 | 0/1 |
| pydata/xarray | 1/2 | 0/2 |
| pylint-dev/pylint | 0/1 | 0/1 |
| pytest-dev/pytest | 0/2 | 0/2 |
| scikit-learn/scikit-learn | 2/3 | 0/3 |
| sphinx-doc/sphinx | 0/5 | 0/5 |
| sympy/sympy | 3/8 | 1/8 |

---

## ADJUDICATION — CPU-only, banked 2026-07-06 (serving verified → verdict VALID)

**VERDICT: stock-AR 19/50 vs diffusion 2/50, net −17, McNemar p ≈ 0, PARITY = FALSE.** The certified
RL-v2 hybrid-clean twin is **NOT at SWE-bench-Verified resolve-parity** with stock AR at N=50 diverse.
Detectable-effect statement (pre-registered |net|>2) satisfied: **|−17| ≫ 2.** Both diffusion wins
(sympy-16886, django-14373) were also AR wins ⇒ **diffusion-only = 0** (no task the twin wins and AR
loses). Pool hash `fe1973937dfb500b…` == frozen manifest (n=50); paired McNemar valid.

### (1) SERVING-HEALTH VERIFICATION — CLEAN (per-request, from artifacts)
- **decode_mode=hybrid_clean CONFIRMED, not canvas-silent-serve.** Serve banner
  `policy=hybrid_clean decode=hybrid_clean apc=1 block=1024 canvas=32 flare=1 bidir=1` +
  engine state `Qwen3_5FlareModelState ready: decode_mode=hybrid_clean` (`logs/diffusion_server.log`).
  The A1 launcher bug (exports only `FLARE_DECODE_POLICY` → engine silently boots the canvas Gumbel
  sampler with no grammar/FSM path) did **not** occur: `VLLM_QWEN3_5_FLARE_DECODE` was exported and
  consumed.
- **Grammar/tool path genuinely fired.** All **2852/2852** successful completions emitted a
  `FLARE hybrid_clean req=… done` line; **2684** reached `stop_reason=complete_tool_call`, 108
  `max_new_tokens`, 64 `None`; **840** `DiffusionDecoding metrics` lines (denoising steps, canvas
  positions evaluated) prove the diffusion decode actually ran — no AR fallback.
- **Sampling envelope = frozen v3, symmetric.** 100% of **2926** diffusion requests and **2364** AR
  requests carried **temp 0.6 / top_p 0.95 / top_k 20** with per-request incrementing seeds
  (proxy dumps). No presence_penalty (correctly dropped).
- **AR arm symmetric + pure AR.** Same chat_template (qwen3-openai-codex.jinja), tool_call_parser
  (qwen3_xml), reasoning_parser (qwen3), max_model_len 32768, max_num_seqs 4, prefix caching. **0**
  FLARE/decode/canvas lines in the AR server log; served `qwen3.5-9b-ar` = stock `Qwen/Qwen3.5-9B`.
  Only intended differences: gmu 0.85 (AR) vs 0.74 (diffusion) + the FLARE stack.
- **Model = the certified config.** Diffusion served `qwen3.5-9b-fastdllm-rlv2-vllm-bf16` —
  **byte-identical model path to the Tier0 v3 gate** (`runs/stage_c_n5v3_gate/logs/diffusion_server.log`).
  Spot-checked 3 episode dumps + a Tier0 v3 dump: coherent, well-formed `run_shell_command` tool calls
  (sed edits, python bug-repros); the diffusion win produced a correct minimal patch
  (`return "%04d" % self.data.year`).
- **HTTP health.** 200-rate diffusion 2852/2926 (97.5%), AR 2324/2364 (98.3%); 400s present in **both**
  arms (2.5% / 1.7%), no `maximum context length` flood — benign agentic-loop context rejections.
- **Boot-probe anomaly explained + benign.** The 12:26Z probe failed `no_allocation_failure=false`
  because it ran seconds after AR-arm teardown before GPU memory settled; the safety gate correctly
  **SKIPPED** the arm (refusing to serve unconfirmed config). The clean-GPU frozen probe is valid
  (`boot_ok=true, headroom 3640 MiB`), and `diff_launch.sh` relaunched at 12:56Z on a settled GPU
  (preflight clear) → healthy boot (KV cache 152,917 tok, 4.67× concurrency). The measured arm ran on
  the healthy server.
- **CAVEAT (disclosed, non-invalidating).** The pre-run plan paragraph named the diffusion arm as the
  b1000 *stock-conversion*; the frozen serve script + v3 gate cert + this run all used the **rlv2**
  twin. The run matches its own certification, so serving is valid — but this is stock-AR vs
  **RL-v2-diffusion** (v3-consistent), not the pristine same-weights paradigm test. The twin carried an
  agentic-RL advantage and still lost 2 vs 19.

### (2) FAILURE SIGNATURE — loop-before-edit on unfamiliar repos
- **Exit taxonomy (diffusion 50):** 26 loop-detector (exit 1) / 13 turn-limit (exit 53) / 10 clean
  (exit 0) / 1 budget (exit 55). **AR 50:** 26 clean / 24 turn-limit / **0 loop** / 0 budget.
- **35/50 diffusion empty patches** (AR 4): by exit — 18 loop-halt + 8 clean-no-edit + 8 turn-limit +
  1 budget.
- **Loop-halt × edit cross-tab:** of the 26 loop-halts, **18 produced no patch at all**, 8 non-empty,
  1 resolved; **25/26 ended unresolved** (report `halt_resolve_split`: pre-resolve 25, post-resolve 1).
  On Tier0 the twin's loops mostly fired **after** an edit existed; on the diverse pool they fire
  **before** → the model grinds coherent-but-repetitive tool calls until the loop-detector halts it
  pre-commit. Median empty-patch agent-wall **673 s** (incl. retry) vs AR 229 s; all-episode median
  wall 535 s vs 115 s.
- **Per-repo:** the gap is broad, worst on the largest/most-diverse buckets — django **11/22 → 1/22**,
  matplotlib 2/3 → 0/3, sklearn 2/3 → 0/3, sympy 3/8 → 1/8; the hard repos (astropy, sphinx, pytest,
  xarray, seaborn, pylint, requests) are ~0 for both.
- **Throughput (secondary, honest):** AR **99.6** vs diffusion **21.4 eps/GPU-h** (4.65× slower); the
  long empty-patch grinds drive most of the wall gap.

### (3) STRATEGIC CONSEQUENCE — the fork
- **The SWE-tuning-campaign premise RETURNS, now with a powered verdict, not a hunch.** The N=50 result
  says the general-agentic RL-v2 twin does not transfer to SWE resolve — the missing ingredient is
  **SWE-trajectory training**, not more decode engineering. The certified loop already produces the
  data (data-gen yield 0.25 is GO-priced) and the serving spine (hybrid-clean, verified clean here).
  **Recommended primary fork: train the base on SWE trajectories via the certified loop** (SFT on
  successful AR/expert SWE episodes ± SWE-reward RL), then re-run W2 N=50 as the acceptance gate.
- **Engineering track (parallel, cheaper): decode-policy work on the loop pattern.** 26/50 loop-halts,
  25 pre-edit — the loop-detector is halting a model that never commits. Levers: earlier/edit-biased
  planning prompts, loop-detector tuning that distinguishes exploration from stall, forced-edit budget,
  retry policy on empty patch. This is a symptom mitigation, not the cure; the cure is training.
- **Do NOT re-litigate serving.** Serving is certified-clean and matches the v3 cert; further N=50
  reruns on the *same* rlv2 twin will reproduce ~2/50. The next model must differ (SWE-tuned).

**Bank status:** verdict banked here + in `swe_endgoal_plan.md` (Stage-C status) + `REPRODUCE_V3.md`
(§0 IS-NOT). Serving verified from artifacts; verdict VALID.
