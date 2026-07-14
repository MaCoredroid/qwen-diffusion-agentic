#!/usr/bin/env python3
"""Consolidate the W-1c LiveCert results into results.json + a summary print."""
import json

R = "/home/mark/qwen_diffusion/runs/w1c_livecert"
det_off = json.load(open(f"{R}/det_gateoff.json"))
det_on = json.load(open(f"{R}/det_gateon.json"))

det = []
for o, n in zip(det_off, det_on):
    det.append({"idx": o["idx"], "path": o["path"],
                "gateoff_det": o["distinct"] == 1, "gateoff_exact": f"{o['exact']}/5",
                "gateon_det": n["distinct"] == 1, "gateon_exact": f"{n['exact']}/5",
                "gateon_corrupts": n["exact"] < 5})
copy_corrupt = sum(1 for d in det if d["gateon_corrupts"])
off_exact_tot = sum(int(d["gateoff_exact"].split("/")[0]) for d in det)
on_exact_tot = sum(int(d["gateon_exact"].split("/")[0]) for d in det)

out = {
    "rung": "W-1c OWED LiveCert battery",
    "object": "twin@plain twinK1, FLARE hybrid_clean, mask 248077, maxlen 32768, gmu 0.74, "
              "BIDIR_PROBE=1; gate VLLM_FASTDLLM_W1_DRAFT_VERIFY toggled OFF(ref)/ON",
    "a_fa_battery": {
        "n_cases": 12, "deploy_cases": 6,
        "deploy_full_span_false_accepts": 0, "byte_identical_on_vs_off": "12/12",
        "on_emit_eq_canonical": "12/12",
        "BAR": "deploy-class full-span false-accepts = 0",
        "registered_battery_PASS": True,
        "note": "registered cases seed BOTH canonical + distractor (forces common-prefix "
                "guard); they did not exercise the single-source copy-value failure mode (b) found",
    },
    "b_throughput": {
        "copy_heavy": {"gateoff_K1_tok_per_fwd": 1.191, "gateon_W1_tok_per_fwd": 2.798,
                       "speedup_x": 2.35, "gateoff_ms_per_committed_tok": 24.35,
                       "gateon_ms_per_committed_tok": 15.42, "reject_tax_share": 0.859,
                       "fired": "6/6"},
        "path": {"gateoff_K1_tok_per_fwd": 1.539, "gateon_W1_tok_per_fwd": 1.539,
                 "gateoff_ms_per_committed_tok": 20.68, "gateon_ms_per_committed_tok": 26.66,
                 "reject_tax_share": 1.0, "fired": "0/4",
                 "note": "non-copy turns net SLOWER: wasted verify forwards"},
        "blended": {"gateoff_K1_tok_per_fwd": 1.243, "gateon_W1_tok_per_fwd": 2.427,
                    "speedup_x": 1.95, "gateoff_ms_per_committed_tok": 23.67,
                    "gateon_ms_per_committed_tok": 17.52},
        "vs_cpu_cert": {"cpu_cert_copy_tok_per_fwd": 14.41, "live_copy_tok_per_fwd": 2.798,
                        "cpu_cert_blended_speedup": 1.863, "live_blended_speedup": 1.95,
                        "reading": "live copy tok/fwd (2.80) is ~5x BELOW the CPU-cert 14.41 "
                                   "projection: the strict whole-span full-reveal verify rejects "
                                   "~86% of drafts on-policy (the teacher-forced abar=0.9913 "
                                   "projection does not hold live)"},
    },
    "c_a6": {
        "n": 5, "fired": 5, "unfired": 0,
        "fired_toolcall_byte_identical": "2/2", "fired_toolcall_args_exact": "2/2",
        "free_text_diverged": "1/3 content + reasoning drift on 2/3",
        "unfired_byte_identical_evidence": "4/4 path turns in (b) (0 firing) byte-identical",
        "note": "every REAL agentic turn fires; tool-call arg turns exact on this sample, "
                "but (b) proves fired copy-value turns can corrupt -> exact-args NOT guaranteed",
    },
    "d_c46_ab": {
        "status": "NOT RUN",
        "reason": "the byte-lossless precondition (structural FA=0 / K=1-equal on copy mass) "
                  "FAILED at (b); a throughput A/B of a value-corrupting fast path is not a "
                  "meaningful win and would inject corrupted edits into episodes. Disciplined "
                  "stop; GPU/budget preserved. Re-owed once the seam is fixed.",
    },
    "DECISIVE_determinism_control": {
        "per_snippet": det,
        "gateoff_baseline": f"deterministic + exact {off_exact_tot}/30 (6/6 snippets 5/5)",
        "gateon_W1": f"exact {on_exact_tot}/30; {copy_corrupt}/6 snippets CORRUPT copied values "
                     f"(idx3 4/5, idx4 4/5, idx5 0/5)",
        "idx5_accept_side": "1 span (31 tok) committed; value '+00:00' -> '+00:' (dropped 00'); "
                            "projected_value_tokens_exact tripwire = 0 (MISSED it)",
        "conclusion": "W-1 FIRING introduces non-deterministic value corruption on live copy "
                      "mass; the CPU-certified structural-FA=0 does NOT transfer to the live "
                      "engine seam (_hc_stage_verify/_hc_verify_read). Baseline K=1 is exact.",
    },
    "VERDICT": "STOP",
    "verdict_reason": "gate-ON W-1 fast path is NOT byte-lossless on live copy mass: it "
                      "non-deterministically corrupts tool-call VALUES (idx5 100%, ~23% of "
                      "copy-heavy generations) that gate-OFF K=1 emits exactly + deterministically. "
                      "Not dispatch-ready for the full C46-under-new-envelope run.",
}
open(f"{R}/results.json", "w").write(json.dumps(out, indent=1))
print(json.dumps({"VERDICT": out["VERDICT"],
                  "a_deploy_FA": out["a_fa_battery"]["deploy_full_span_false_accepts"],
                  "b_blended_speedup": out["b_throughput"]["blended"]["speedup_x"],
                  "b_copy_tok_per_fwd_live_vs_cert": [2.798, 14.41],
                  "byte_safety_gateoff_exact": f"{off_exact_tot}/30",
                  "byte_safety_gateon_exact": f"{on_exact_tot}/30",
                  "snippets_corrupted_gateon": f"{copy_corrupt}/6",
                  "d_status": out["d_c46_ab"]["status"]}, indent=1))
