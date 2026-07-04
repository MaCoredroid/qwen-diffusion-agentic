#!/usr/bin/env python3
"""GROUP-2 gate: the hybrid-clean tool-call FSM, driven off-GPU end-to-end.

The reference FSM decode policy lives in the vLLM pin at
``vllm/v1/sample/hybrid_clean.py`` (``HybridCleanDecodePolicy`` +
``parse_hybrid_clean_request``). The build note flags it as an *orphaned FSM*:
imported by nothing on the serving path, so its "FSM drives bulk commits /
values sequential / zero value projection" guarantees were never actually
exercised by anything runnable.

This gate closes that off-GPU: it imports the REAL module from the pin workspace
(by file path -- the module deliberately imports nothing from torch or vLLM, so
no CUDA/model init is triggered) and DRIVES the policy with a self-contained
char-level tokenizer + scripted oracle. It asserts, on the actual engine code:

  1. fsm_bulk_commit        -- truly-forced grammar tokens are committed in bulk
                               with ZERO model forwards (forwards < generated).
  2. values_sequential      -- exactly one model forward per model-chosen token
                               (forwards == value + structural tokens).
  3. byte_reproduces_target -- the decoded output equals the canonical tool call.
  4. zero_value_projection  -- a well-formed value decode leaves the grammar
                               value-neutral (value_projection_events == 0) and
                               verify_invariants() passes.
  5. projection_counter_live-- a MIS-ROUTED value position (the documented
                               failure mode) makes value_projection_events fire
                               and trips verify_invariants(): the counter is a
                               live tripwire, not a tautological 0.
  6. request_wiring_composes-- the SamplingParams.extra_args ->
                               parse_hybrid_clean_request -> build_grammar ->
                               HybridCleanDecodePolicy.generate path drives the
                               decode from a request payload alone.

This is a promotion-gate instrument (like ``parity_audit_flare_engine.py
--mode ops-parity``), not the serving path: wiring the FSM into the batched GPU
``Qwen3_5FlareSampler`` canvas path remains GPU-only work. But it makes the FSM
a genuinely-driven, gated component rather than dead reference code.

Exit codes: 0 = all gates passed; 2 = a gate FAILED; 3 = the FSM module could
not be located in the pin workspace.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_VLLM_WORKSPACE = Path("/home/mark/shared/vllm_p2_pr42406")

VOCAB_SIZE = 128

SCHEMAS_ONE = {"get_weather": {"properties": {"location": {"type": "string"}}}}
CANONICAL_ONE = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=location>\n"
    "Paris\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


class CharTokenizer:
    """1 token == 1 code point, so grammar string reasoning maps onto ids."""

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(t)) for t in token_ids if 0 <= int(t) < VOCAB_SIZE)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in str(text)]


class OracleModel:
    """Next-token logits that argmax to the next char of ``target``."""

    def __init__(self, tokenizer, target: str):
        self.tokenizer = tokenizer
        self.target = target
        self.calls = 0

    def __call__(self, committed):
        self.calls += 1
        text = self.tokenizer.decode(committed)
        if not self.target.startswith(text):
            raise AssertionError(
                f"decoded prefix diverged from target: {text!r} vs {self.target!r}"
            )
        logits = [0.0] * VOCAB_SIZE
        if len(text) < len(self.target):
            logits[ord(self.target[len(text)])] = 100.0
        return logits


def _load_hybrid_clean(workspace: Path):
    """Import the REAL ``hybrid_clean`` module from the pin workspace file."""
    path = Path(workspace) / "vllm" / "v1" / "sample" / "hybrid_clean.py"
    if not path.exists():
        raise FileNotFoundError(f"hybrid_clean FSM module not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "_hybrid_clean_standalone", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load hybrid_clean from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def run_gate(workspace: Path) -> dict[str, Any]:
    try:
        hc, module_path = _load_hybrid_clean(workspace)
    except (FileNotFoundError, ImportError) as exc:
        return {
            "gate": "fsm-hybrid-clean",
            "module_available": False,
            "error": str(exc),
            "passed": None,
            "verdict": "FSM_MODULE_UNAVAILABLE",
        }

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    tok = CharTokenizer()

    # --- clean decode: bulk commit + sequential values + zero projection ---
    grammar = hc.HybridCleanGrammar(tok, SCHEMAS_ONE, grammar_topk=64)
    policy = hc.HybridCleanDecodePolicy(tok, grammar, max_new_tokens=512)
    model = OracleModel(tok, CANONICAL_ONE)
    out_ids, stats = policy.generate(model)

    record(
        "byte_reproduces_target",
        tok.decode(out_ids) == CANONICAL_ONE,
        f"stop_reason={stats.stop_reason}",
    )
    record(
        "fsm_bulk_commit",
        stats.fsm_committed_tokens > 0 and stats.forwards < stats.generated_tokens,
        f"forced={stats.fsm_committed_tokens} forwards={stats.forwards} "
        f"generated={stats.generated_tokens}",
    )
    record(
        "values_sequential",
        stats.forwards == stats.model_chosen_tokens
        and model.calls == stats.forwards
        and stats.value_tokens > 0,
        f"forwards={stats.forwards} model_chosen={stats.model_chosen_tokens} "
        f"value_tokens={stats.value_tokens} model_calls={model.calls}",
    )
    clean_ok = stats.value_projection_events == 0
    try:
        stats.verify_invariants()
    except AssertionError as exc:
        clean_ok = False
        clean_detail = f"verify_invariants raised: {exc}"
    else:
        clean_detail = f"value_projection_events={stats.value_projection_events}"
    record("zero_value_projection_clean", clean_ok, clean_detail)

    # --- projection counter is LIVE: mis-routed value fork must trip it ---
    grammar2 = hc.HybridCleanGrammar(tok, SCHEMAS_ONE, grammar_topk=64)
    fork_prefix = "<tool_call>\n<function=get_weather>\n<"  # 'p' vs '/' fork
    real_inside = grammar2.inside_value
    # The phase detector MIS-REPORTS the structural fork as inside-a-value.
    grammar2.inside_value = (
        lambda text: text == fork_prefix or real_inside(text)
    )

    class ForkIllegalModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, committed):
            self.calls += 1
            text = tok.decode(committed)
            logits = [0.0] * VOCAB_SIZE
            if text == fork_prefix:
                logits[ord("X")] = 100.0  # illegal top -> grammar must steer
            elif len(text) < len(fork_prefix):
                logits[ord(fork_prefix[len(text)])] = 100.0
            else:
                logits[ord("/")] = 100.0
            return logits

    policy2 = hc.HybridCleanDecodePolicy(
        tok, grammar2, max_new_tokens=len(fork_prefix) + 1
    )
    _out2, stats2 = policy2.generate(ForkIllegalModel())
    tripwire_fired = stats2.value_projection_events > 0
    try:
        stats2.verify_invariants()
    except AssertionError:
        invariants_caught = True
    else:
        invariants_caught = False
    record(
        "projection_counter_live",
        tripwire_fired and invariants_caught,
        f"value_projection_events={stats2.value_projection_events} "
        f"invariants_caught={invariants_caught}",
    )

    # --- request -> policy composition (the flagged unwired entry points) ---
    sampling_params = SimpleNamespace(
        extra_args={
            "decode_policy": "hybrid_clean",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "properties": {"location": {"type": "string"}}
                        },
                    },
                }
            ],
            "grammar_topk": 64,
        }
    )
    config = hc.parse_hybrid_clean_request(sampling_params)
    wiring_ok = config is not None
    if wiring_ok:
        gram3 = config.build_grammar(tok)
        policy3 = hc.HybridCleanDecodePolicy(tok, gram3, max_new_tokens=512)
        out3, stats3 = policy3.generate(OracleModel(tok, CANONICAL_ONE))
        wiring_ok = (
            tok.decode(out3) == CANONICAL_ONE
            and stats3.fsm_committed_tokens > 0
            and stats3.forwards == stats3.model_chosen_tokens
            and stats3.value_projection_events == 0
        )
    record(
        "request_wiring_composes",
        wiring_ok,
        "parse_hybrid_clean_request -> build_grammar -> generate",
    )

    passed = all(c["ok"] for c in checks)
    return {
        "gate": "fsm-hybrid-clean",
        "module_available": True,
        "module_path": str(module_path),
        "vllm_workspace": str(workspace),
        "checks": checks,
        "num_checks": len(checks),
        "num_failed": sum(1 for c in checks if not c["ok"]),
        "passed": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vllm-workspace", type=Path, default=DEFAULT_VLLM_WORKSPACE
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report = run_gate(args.vllm_workspace)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")

    passed = report.get("passed")
    if passed is None:
        return 3
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
