# DFlash-style Diffusion Drafter for our AR-9B — Decision Plan (workflow w7t45sy49, 2026-07-03)

12 agents, 0 errors; claims verified vs papers/code/local configs.

# Decision Plan: Small Diffusion Drafter (DFlash-style) for Our AR-9B Target on One RTX 5090

Verified locally before writing: our B@1000 export (`/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-b1000-vllm-bf16/config.json`) has vocab_size **248,320**, hidden 4096, 32 layers (GDN hybrid, full-attn every 4th layer) — **architecturally exact match** to z-lab/Qwen3.5-9B-DFlash (same vocab, same num_target_layers=32, taps [1,5,9,13,17,21,25,29]; taps land mostly on GDN layers, which is fine — feature extraction reads layer outputs, attention-type-agnostic). The `151,936 x 4B` lm_head constraint in our notes is stale Qwen3-era; the real transient is `block x 248,320 x 4B`. Disk has 3.2T free → offline feature caching is viable at pilot scale. The p2 venv has transformers 5.12.1 (parses the drafter's new-style `rope_parameters`). Also noticed: our target config carries `mtp_num_hidden_layers: 1` — Qwen3.5-9B ships a native MTP head, which if weights survived our merge is a zero-training spec-decode comparator we should measure in M0.

---

## 1. FEASIBILITY VERDICT: YES, conditionally — trainable and servable on one 5090, with the agentic-content acceptance number genuinely unknown

**Drafter**: adopt the z-lab Qwen3.5-9B-DFlash topology verbatim — 6 Qwen3 layers (5 SWA-4096 + 1 full), hidden 4096, 1.29B trainable params (2.58GB bf16), embedding+LM head shared frozen with the target (no extra 2GB), block 16, 8 target-layer taps → W_c → RMSNorm → injected as extra KV in every draft layer. The off-the-shelf checkpoint loads against our target class directly; it was trained against **stock** Qwen3.5-9B, our B@1000 merge shifts hidden states by an unknown amount — measuring that delta-tau is the first experiment, then we retrain (retrain-freely rule; note the drafter is target-frozen, so **capability-erosion constraints do not apply** — drafter quality is tau and nothing else, our hardest training constraint vanishes here).

**Training on 32GB**: two viable paths.
- *Offline features* (preferred for GPU util): cache 5–8 tap layers to disk at 40–64KB/token; 5–20K samples × ~1.5K tok = 8–30M tokens = 0.3–1.9TB — fits our 3.2T disk. Then drafter-only training: 2.58GB weights + 2.58GB grads + 8-bit Adam ~2.6GB + activations → comfortable, batch 2–4 at seq 1536–3072, fully util-compliant in both phases.
- *In-loop bf16 target forward*: 18GB target + ~8GB drafter stack — fits at batch 1, tight; only if disk churn becomes a problem.
- Budget: response regeneration by our own target (recipe requires it) ~2–4 GPU-h per 5K samples on vLLM; training 5K×5 epochs ≈ order 10–20 GPU-h. Paper-scale 800K samples is out and not needed for a verdict — speculators' 5K-sample run already reached tau ~5.9 on chat.

**Serving on sm_120**: three routes, recommend C.
- (A) SGLang: the officially supported DFlash path, but its reference backends (fa4, trtllm_mha, flashinfer) are datacenter-Blackwell; abandons our vLLM assets. Rejected as primary.
- (B) vLLM PR #40898 (unmerged): reference implementation to crib from, but standard-attention spec decode — **variable accept on a GDN hybrid target requires SSM-state scatter it likely lacks**.
- (C) **Our vllm-main pin (#42406)**: already contains exactly the missing primitive — MRV2 ModelState, canvas-as-DRAFT-TOKENS commit path, variable-accept in-kernel GDN state scatter (`num_accepted_tokens` + `ssm_state_indices`), per-request causal/bidirectional Triton attention, align APC. The DFlash drafter becomes a ModelState riding the DiffusionGemma spec-decode path. This is the paused P2 work, resurrected with a drafter that has published acceptance numbers instead of our failed self-spec.
- Serve-time memory: 18GB target + 2.6GB drafter + ~42MB/req feature cache + KV + fp32 SSM cache → fine standalone (util ~0.9); **does not fit the 0.745 co-run budget** — spec-decode serving is a solo-GPU activity.

**Honest expected numbers** (per promotion discipline, none of these count until measured on our eval, our engine, our card):
- *(a) Reasoning/GSM8K-class*: paper tau 6.5–7.9 greedy on H200/B200; tau should mostly transfer, wall-clock will not (5090 bandwidth, enforce-eager/no CUDA graphs makes the 6-layer draft forward relatively pricier). Budget **tau 4.5–6.5, wall-clock 2–3.5x** at concurrency 1. The ≥2x gate is plausibly clearable here.
- *(b) STRICT tool-call content*: **unknown — no published source measures it.** Nearest proxies are all DFlash's weakest tier (MT-Bench tau 4.24/2.75x, SWE-Bench 2.92x, Alpaca 2.27x, datacenter hardware). Two of our own facts cut opposite ways: (i) the factorization barrier survives — one denoising step per block = independent marginals within the block, so C>0 paired-value spans stay draft-hostile and tau on exact-argument runs will crater toward 1–2; but (ii) unlike our monolithic canvas, DFlash **re-conditions every round** — block token 1 always conditions on all committed tokens, and rejection truncates at first divergence — so the barrier binds only within 16 tokens and costs *tau, never exactness* (all accepted tokens are verifier-sanctioned; the loop is lossless by construction). Our left-context-first result (0/41 → 40/41 once keys commit before values) predicts the verify loop's natural key-then-value round structure recovers much of the ordering benefit for free. Budget **overall agentic tau 2–3.5, wall-clock 1.5–2.5x** — report as unknown-until-M1.
- *Grammar-constrained drafting* (our unique layer, nobody else will build it): project drafter logits through the wave-1 structural FSM so proposed scaffold tokens are always grammar-valid (raises structural acceptance, still lossless — drafts are only proposals, the verifier sanctions); additionally, truly-forced tokens (grammar admits exactly one) can be committed without verify as the label-free **constrained lane** — permitted under promotion discipline provided value force-counters stay 0 and forwards/turn is audited. This converts the drafter's capacity budget to content tokens only.
- *DSpark-style serial Markov head*: nothing in our verified ground covers DSpark — treat it as an unverified M4 variant (a lightweight serial head over drafter states to model intra-block conditionals) to try **only if** M1 shows value-span tau is the binding constraint. Do not plan numbers around it.

## 2. REUSE MAP

| Our asset | Slot in this build |
|---|---|
| B@1000 vLLM export + AR-parity gate | The target, unchanged; parity gate doubles verbatim as the spec-decode losslessness gate |
| vllm-main pin #42406 + p2 plan seam table | Serving host; MRV2/DiffusionGemma/variable-accept-GDN-scatter is the exact engine primitive; M1 smoke gauntlet transfers |
| .venv-vllm 0.23 + FR13 align APC | Baseline/fallback arm; guided-AR baseline must be **re-measured on the pin build** before any speedup claim (R6) |
| flare_hf_cache.py port spec | Spec for the drafter's GDN-boundary/conv-tail state wiring; the drafter itself is pure attention so mostly needed target-side |
| Grammar decoder + per-call waves + schema_aware_drafter FSM seeds | Drafter-side structural projection (M3); wave schedule informs proposal ordering |
| Run-1 5055 mix, 398-ep leak-checked RL pool, rl_dataset_plan splits | Prompt sources for target-regenerated drafter training data (qwen3_xml native format, full 248,320 vocab — **no draft-vocab truncation**, it would drop tool tokens) |
| Hashed matched-20 + never-train BFCL/API-Bank slices | Acceptance-rate eval per content class; never trained on |
| Forwards/turn monitor, force-counters, generated-token audit, honest scoreboard | Losslessness + contamination audits for the spec loop |
| fastdllm two-stream scaffold | *Mostly not needed* — drafter is trained fresh from the paper recipe (via speculators), not converted; scaffold only if we init drafter layers from target layers |
| 27B SGLang / GB10 flywheel | Later: drafter for a stronger target; flywheel = agentic acceptance eval harness |

## 3. MILESTONES

**M0 — Off-the-shelf measurement (≤10 GPU-h, ~2 days).** Load z-lab drafter + stock Qwen3.5-9B via their Transformers-backend code on sm_120; reproduce tau≥5 on GSM8K slice. Then swap in OUR B@1000 target and measure delta-tau (distribution-shift cost). Measure tau per content class on the never-train BFCL slice. Memory co-residency test. Check whether MTP weights survived our merge (free comparator). Feature-fidelity probe: tau with NF4-extracted vs bf16-extracted target features. **Gate**: stock reproduction within ~20% of paper tau. **Kill**: sm_120/transformers blockers exceeding ~2 days of fixes.

**M1 — Drafter retraining on our traces (~25–45 GPU-h, ~1 wk).** Regenerate 5–20K responses with our target (native qwen3_xml, leak-checked prompts, disjoint from eval); offline feature cache; train per paper recipe (anchor masking, w_k=exp(-(k-1)/γ), block 16 trained = block 16 deployed). **Gates**: held-out reasoning tau ≥4; tool-call tau measured and reported per span class; losslessness audit clean. **Kill (K3-analog)**: trained tau on our agentic traces <2.5 overall → **stop before any engine work** — the acceptance side didn't materialize and no engine integration can fix it.

**M2 — Pin integration + wall-clock (~15–25 GPU-h + engineering, 1–2 wks).** DFlash-as-ModelState on #42406; target-side hidden-state taps; variable-accept GDN scatter; APC co-enablement; re-baseline guided-AR on the same build. **Gates**: byte-identical greedy vs AR baseline (parity suite); ≥2x wall-clock on reasoning @ conc 1; ≥1.3x on full agentic turns; util profile clean (no .item()-class syncs). **Kill**: <1.2x net agentic after profiling → stop, keep 0.23 route, write postmortem.

**M3 — Grammar-integrated drafting (~10 GPU-h).** FSM structural projection on drafter logits + truly-forced structural commits. **Gates**: exact-args ≥ AR-guided 50/63; value force-counters 0; measurable tool-call-turn speedup over M2.

**M4 (conditional) — DSpark-style serial head experiment**, only if M1/M3 show value-span tau binding. Total: ~70–120 GPU-h over 3–5 weeks, sequenced after/around the live RL-v4 pilot (one GPU).

## 4. RELATION TO THE CLOSED SPEED CAMPAIGN — what's actually different

**Genuinely different**: (1) drafter cost — 6 layers/1.3B vs full 9B kills the 21x-slower draft forward, so rejected rounds are cheap; (2) the drafter is *trained for acceptance* against target continuations with position-decayed loss — our self-spec drafter never was, and the SDTT failure explicitly said "train a separate small drafter, don't compress the 9B into its own" — this is that; (3) target-hidden-state KV injection gives the drafter information our standalone model never had; (4) per-round re-conditioning bounds the factorization barrier to 16 tokens and converts it from an exactness problem to a tau discount; (5) engine-side, we're reusing P2 plumbing that already exists, not building it.

**NOT different — be honest**: (1) the factorization barrier itself is untouched — one parallel forward is still independent marginals, C>0 spans will still cap tau, and DFlash's own numbers confirm chat/agentic is its weakest tier; (2) tool-call content remains unmeasured by everyone including z-lab — we are extrapolating from MT-Bench again, exactly the trap the promotion discipline exists for; (3) all headline speedups are H200/B200 — wall-clock transfer to a 5090 under enforce-eager is unproven; (4) **strategically this is a pivot, not a continuation**: it does not advance the standalone diffusion-conversion thesis one inch — it redeploys the expertise as a lossless serving sidecar for the AR target. Its wins are engine-lane wins, not raw diffusion-model evidence, and should be scoreboarded as such.

## 5. OPEN QUESTIONS ONLY EXPERIMENTS ANSWER

1. Delta-tau of the stock z-lab drafter against our fine-tuned B@1000 target (distribution shift cost) — M0.
2. tau on strict qwen3_xml spans, split scaffold / C~0 values / C>0 paired values — no published data exists anywhere.
3. Does drafter-side FSM projection raise structural tau without contaminating (force-counter + forwards/turn audits clean)?
4. 5090 draft-forward : verify-forward wall-clock ratio under enforce-eager — determines whether tau converts to speedup at all (paper shows architectures where tau 3.9 → only 1.2x).
5. Does variable-accept SSM scatter compose with FR13 align-mode APC (state consistency across accept/reject + prefix reuse)?
6. NF4-extracted feature fidelity: does quantized-target extraction cost tau (decides the training memory plan)?
7. Minimum training data for agentic tau (5K vs 20K vs 50K target-regenerated samples) — erosion doesn't constrain us here, only tau-vs-GPU-hours.
8. Multi-turn regime economics: per-turn feature-cache rebuild + short generation turns between tool-result prefills — unmeasured in every source, and it is 92% of our turn time.
9. Whether the native MTP head in our export is live weight — a possibly-free spec-decode baseline that would reset the bar M2 must beat.

**Recommendation**: greenlight M0 immediately (it's ~2 days, mostly measurement, and question 1 alone is worth it), hold M2 engine work behind the M1 tau gate, and sequence GPU time after the RL-v4 matched-20 readout.