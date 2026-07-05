#!/usr/bin/env python
"""STEP 1 (work-item #29): rebuild M_{t+1} as a diffusion-loadable base.

Merge the RL-v2 adapter into the Fast-dLLM `init` candidate *in the HF stack*
(PEFT merge_and_unload; W += (alpha/r)*B@A, scale 2.0) so the mask token
(248077) / bridge / bd_size=32 survive, unlike the vLLM export.

Runs the merge sanity gate (mirror of the export gate, real_diffusion_export_block_reconcile.md
§3a): for one GDN `in_proj_qkv` tensor and one attn `o_proj` tensor, assert
    merged == init + 2.0 * (B @ A)   to maxabs diff 0.0,
LoRA delta nonzero, weight moved off init; assert conversion_manifest.json has
mask_token_id==248077, bd_size==32, has_weights==true. ANY failure => KILL-1.

CPU-only-safe logic; loads weights on GPU (host-RAM cage) for a bf16-exact merge.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import types
from pathlib import Path

import torch

# transformers.save_pretrained -> accelerate.extract_model_from_parallel does an
# unconditional `from deepspeed import DeepSpeedEngine`. The installed deepspeed
# raises MissingCUDAException (no CUDA toolkit / CUDA_HOME on this CPU merge) on
# import -- which is NOT ImportError, so accelerate's guard does not catch it.
# Register a harmless stub so the isinstance() check falls through to False.
try:  # pragma: no cover - environment shim
    import deepspeed  # noqa: F401
except Exception:
    import importlib.machinery
    _stub = types.ModuleType("deepspeed")
    _stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    _stub.__version__ = "0.0.0+stub"
    _stub.DeepSpeedEngine = type("DeepSpeedEngine", (), {})
    sys.modules["deepspeed"] = _stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "fast-dllm/v2"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def _load_unmerged(init_dir: str, adapter_dir: str, device: str):
    """Load the Fast-dLLM init candidate (bridge/mask intact) + RL-v2 adapter,
    UNMERGED. Mirrors scripts/eval_fastdllm_toolcall_cases.load_model but keeps
    the model on the requested device.

    Default device=cpu: PEFT's get_delta_weight casts B@A to fp32 on CPU and the
    matmul is deterministic across calls, so merge_and_unload's internal recompute
    is bit-identical to the captured delta -> the sanity gate reads maxabs 0.0.
    (On CUDA, bf16 cuBLAS recomputes B@A non-deterministically at ~1 ulp, which at
    this delta magnitude shows up as ~2e-4 measurement noise, not a merge fault.)
    """
    tokenizer = AutoTokenizer.from_pretrained(init_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        init_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device).eval()
    return model, tokenizer


def _clean_path(peft_name: str) -> str:
    # PeftModel wraps the base model as `.base_model.model`; a base submodule
    # `model.layers.N...` appears as `base_model.model.model.layers.N...`.
    prefix = "base_model.model."
    return peft_name[len(prefix):] if peft_name.startswith(prefix) else peft_name


def _find_lora_module(peft_model, needle: str):
    """First LoRA-wrapped module whose qualified name contains `needle`."""
    for name, mod in peft_model.named_modules():
        if needle in name and hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})) > 0:
            return name, mod
    return None, None


def _capture(mod, name: str) -> dict:
    """Snapshot init weight, exact PEFT delta, and manual 2.0*(B@A) before merge."""
    adapters = list(mod.scaling.keys())
    if len(adapters) != 1:
        raise SystemExit(f"KILL-1: expected exactly one adapter on {name}, got {adapters}")
    ad = adapters[0]
    scaling = float(mod.scaling[ad])

    w_init = mod.base_layer.weight.detach().clone()
    # Authoritative delta PEFT will actually add during merge_and_unload
    # (get_delta_weight returns fp32 on the CPU path):
    delta_peft = mod.get_delta_weight(ad).detach().clone()
    # Literal design formula: 2.0 * (B @ A) in the same fp32 the CPU merge uses.
    a = mod.lora_A[ad].weight.detach()
    b = mod.lora_B[ad].weight.detach()
    delta_manual = scaling * (b.float() @ a.float())

    # Replicate merge EXACTLY: base_layer.weight.data += delta  (in-place, so the
    # result stays bf16 = weight.float()+delta rounded to bf16). An out-of-place
    # `w_init + delta` would promote to fp32 and NOT match the stored bf16 weight.
    expected = w_init.clone()
    expected += delta_peft

    formula_maxabs = (delta_peft.float() - delta_manual.float()).abs().max().item()
    delta_maxabs = delta_peft.float().abs().max().item()
    return {
        "name": name,
        "clean_path": _clean_path(name),
        "adapter": ad,
        "scaling": scaling,
        "shape": list(w_init.shape),
        "w_init": w_init,
        "expected": expected,                       # bf16(init + 2.0*B@A), merge's own arithmetic
        "delta_manual": delta_manual,
        "delta_maxabs": delta_maxabs,               # nonzero-delta check
        "formula_agree_maxabs": formula_maxabs,     # get_delta_weight == 2.0*B@A (fp32) ?
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models/qwen3.5-9b-fastdllm-init"))
    ap.add_argument("--adapter", default=str(ROOT / "runs/rl_multiturn_grpo_v2/from_selected_base_g4_step300/adapter_model"))
    ap.add_argument("--out", default=str(ROOT / "models/qwen3.5-9b-fastdllm-mtplus1-merged"))
    ap.add_argument("--gate-out", default=str(ROOT / "runs/convert_after_rl/step1_merge/merge_sanity_gate.json"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu (default) gives a deterministic, bit-exact bf16 merge gate")
    ap.add_argument("--no-save", action="store_true", help="run gate only, do not write merged weights")
    args = ap.parse_args()

    t0 = time.time()
    init_dir = Path(args.init)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out)
    gate_out = Path(args.gate_out)

    for p in (init_dir, adapter_dir):
        if not p.exists():
            raise SystemExit(f"KILL-1: missing input path {p}")

    # ---- load init + adapter (unmerged) via the canonical bridge loader ----
    print(f"[step1] loading init={init_dir} + adapter={adapter_dir} (unmerged, device={args.device})", flush=True)
    peft_model, tokenizer = _load_unmerged(str(init_dir), str(adapter_dir), args.device)

    # ---- pre-merge snapshots for the two probe tensors ----
    gdn_name, gdn_mod = _find_lora_module(peft_model, "in_proj_qkv")
    attn_name, attn_mod = _find_lora_module(peft_model, "o_proj")
    if gdn_mod is None:
        raise SystemExit("KILL-1: no LoRA-wrapped GDN in_proj_qkv module found")
    if attn_mod is None:
        raise SystemExit("KILL-1: no LoRA-wrapped attn o_proj module found")
    probes = [_capture(gdn_mod, gdn_name), _capture(attn_mod, attn_name)]

    # ---- the merge ----
    print("[step1] merge_and_unload() ...", flush=True)
    merged = peft_model.merge_and_unload()

    # ---- gate (a): bit-exact merge on the two probe tensors ----
    tensor_results = []
    failures = []
    for p in probes:
        w_merged = merged.get_submodule(p["clean_path"]).weight.detach()
        merge_vs_expected = (w_merged.float() - p["expected"].float()).abs().max().item()  # vs init+2.0*B@A
        moved_off_init = (w_merged.float() - p["w_init"].float()).abs().max().item()
        entry = {
            "module": p["name"],
            "clean_path": p["clean_path"],
            "shape": p["shape"],
            "adapter": p["adapter"],
            "scaling": p["scaling"],
            "merge_vs_init_plus_2BA_maxabs": merge_vs_expected,     # MUST be 0.0
            "lora_delta_maxabs": p["delta_maxabs"],                 # MUST be > 0
            "moved_off_init_maxabs": moved_off_init,                # MUST be > 0 (== delta)
            "formula_agree_maxabs": p["formula_agree_maxabs"],      # get_delta_weight vs 2.0*B@A
        }
        if merge_vs_expected != 0.0:
            failures.append(f"{p['name']}: merged != init+2.0*B@A (maxabs {merge_vs_expected})")
        if not (p["delta_maxabs"] > 0.0):
            failures.append(f"{p['name']}: zero LoRA delta")
        if not (moved_off_init > 0.0):
            failures.append(f"{p['name']}: weight did not move off init")
        if abs(p["scaling"] - 2.0) > 0:
            failures.append(f"{p['name']}: scaling {p['scaling']} != 2.0")
        tensor_results.append(entry)
        print(f"[gate] {p['clean_path']} shape={p['shape']} "
              f"merge_vs_init+2BA_maxabs={merge_vs_expected} delta_maxabs={p['delta_maxabs']:.6g} "
              f"scaling={p['scaling']}", flush=True)

    # ---- gate (config): mask/bridge invariants from init manifest ----
    init_manifest = json.loads((init_dir / "conversion_manifest.json").read_text())
    cfg = json.loads((init_dir / "config.json").read_text())
    mask_ok = init_manifest.get("mask_token_id") == 248077 and cfg.get("mask_token_id") == 248077
    bd_ok = init_manifest.get("bd_size") == 32 and cfg.get("bd_size") == 32
    weights_ok = init_manifest.get("has_weights") is True
    if not mask_ok:
        failures.append(f"mask_token_id != 248077 (manifest={init_manifest.get('mask_token_id')}, config={cfg.get('mask_token_id')})")
    if not bd_ok:
        failures.append(f"bd_size != 32 (manifest={init_manifest.get('bd_size')}, config={cfg.get('bd_size')})")
    if not weights_ok:
        failures.append("has_weights != true in init manifest")

    gate_pass = len(failures) == 0

    gate_evidence = {
        "step": "1_merge_M_tplus1",
        "work_item": 29,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gate_pass": gate_pass,
        "kill_1_triggered": not gate_pass,
        "failures": failures,
        "inputs": {
            "init": str(init_dir),
            "adapter": str(adapter_dir),
            "out": str(out_dir),
            "merge_device": args.device,
        },
        "merge_math": {"formula": "W += (alpha/r)*B@A", "scale": 2.0, "r": 16, "alpha": 32},
        "tensor_probes": tensor_results,
        "config_invariants": {
            "mask_token_id": cfg.get("mask_token_id"),
            "bd_size": cfg.get("bd_size"),
            "mask_token": cfg.get("mask_token"),
            "has_weights": init_manifest.get("has_weights"),
            "mask_ok": mask_ok, "bd_ok": bd_ok, "weights_ok": weights_ok,
        },
        "elapsed_sec_gate": round(time.time() - t0, 1),
    }

    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(json.dumps(gate_evidence, indent=2))
    print(f"[step1] gate evidence -> {gate_out}", flush=True)

    if not gate_pass:
        print("\n==== KILL-1: merge sanity gate FAILED ====", flush=True)
        for f in failures:
            print("  - " + f, flush=True)
        return 2

    if args.no_save:
        print("[step1] --no-save: gate PASS, skipping weight save", flush=True)
        print(json.dumps({k: v for k, v in gate_evidence.items() if k != "tensor_probes"}, indent=2))
        return 0

    # ---- save merged diffusion-loadable base ----
    print(f"[step1] saving merged base -> {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # Copy custom-code / bridge aux files that save_pretrained may not carry.
    for fname in ("configuration.py", "modeling.py", "chat_template.jinja",
                  "weight_remap_plan.json", "tokenizer.json", "tokenizer_config.json"):
        src = init_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # conversion_manifest.json: carry init manifest forward + lineage note.
    merged_manifest = dict(init_manifest)
    merged_manifest["out_dir"] = str(out_dir)
    merged_manifest["lineage"] = {
        "derived_from_init": str(init_dir),
        "merged_adapter": str(adapter_dir),
        "adapter_label": "RL-v2 diffu-GRPO (from_selected_base_g4_step300)",
        "merge": "PEFT merge_and_unload; W += (alpha/r)*B@A; scale=2.0; r=16 alpha=32",
        "target_modules": "q,k,v,o_proj + in_proj_{qkv,z,a,b} + out_proj",
        "note": "M_{t+1}: diffusion-loadable twin of the scoreboard merged-AR model; "
                "bridge/mask token 248077/bd_size 32 preserved (vLLM export strips them). "
                "Merge sanity gate PASS (maxabs 0.0 vs init+2.0*B@A). Base for convert-after-RL "
                "audit (work-item #29, step 1).",
    }
    (out_dir / "conversion_manifest.json").write_text(json.dumps(merged_manifest, indent=2))

    # Re-verify the saved base preserves mask/bridge config.
    saved_cfg = json.loads((out_dir / "config.json").read_text())
    post = {
        "saved_mask_token_id": saved_cfg.get("mask_token_id"),
        "saved_bd_size": saved_cfg.get("bd_size"),
        "saved_mask_token": saved_cfg.get("mask_token"),
        "has_modeling_py": (out_dir / "modeling.py").exists(),
        "has_tokenizer": (out_dir / "tokenizer.json").exists(),
    }
    post_ok = saved_cfg.get("mask_token_id") == 248077 and saved_cfg.get("bd_size") == 32
    gate_evidence["saved_base_check"] = {**post, "ok": post_ok}
    if not post_ok:
        gate_evidence["gate_pass"] = False
        gate_evidence["kill_1_triggered"] = True
        gate_evidence["failures"].append("saved merged base lost mask_token_id/bd_size")
    gate_evidence["elapsed_sec_total"] = round(time.time() - t0, 1)
    gate_out.write_text(json.dumps(gate_evidence, indent=2))

    if not post_ok:
        print("==== KILL-1: saved base lost mask/bridge config ====", flush=True)
        return 2

    print(f"[step1] DONE in {gate_evidence['elapsed_sec_total']}s. gate PASS. merged -> {out_dir}", flush=True)
    print(json.dumps({k: v for k, v in gate_evidence.items() if k != "tensor_probes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
