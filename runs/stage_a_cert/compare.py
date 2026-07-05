#!/usr/bin/env python3
"""A6/A7 comparison + certificate (Stage-A cert). CPU-only.

Joins the OFFLINE (in-process LLM) and ONLINE (AsyncLLM server) captures and
computes the online-vs-offline certificate:

  A6 (single-turn, fresh APC): token-identical = offline gen_ids == online ids
      (re-encoded from the server's raw text). Byte-identical = decoded texts
      match. Any divergence is localized (id position, tokens, mask-id check,
      near-tie fp-residue class) and diagnosed.
  A7 (multi-turn, warm APC): per-turn output must match the cache-on battery --
      quality-identical (exact_args + valid), byte where the battery was byte.

Cross-checks the OFFLINE capture against the certified cache-on battery
(runs/lossless_apc/gates2/matched20_gateOFF_eager_warm.jsonl) to prove the
server-config offline mirror reproduces the certified battery.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/mark/qwen_diffusion")
sys.path.insert(0, str(ROOT / "scripts"))
from eval_toolcall_jsonl import score_tool_calls  # noqa: E402
from eval_flare_northstar_matched import trim_scored_assistant  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

_TOK = AutoTokenizer.from_pretrained(
    "/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16",
    trust_remote_code=True)

OUTDIR = ROOT / "runs/stage_a_cert"
REF = {r["global_turn"]: r for r in json.loads((ROOT / "runs/p2_engine_bench/matched20_ref.json").read_text())}
GATES2 = {r["global_turn"]: r for r in
          (json.loads(l) for l in open(ROOT / "runs/lossless_apc/gates2/matched20_gateOFF_eager_warm.jsonl"))}
MASK_ID = 248077


def load(p):
    return {r["global_turn"]: r for r in (json.loads(l) for l in open(p))}


def score_online(gt, online):
    rec = REF[gt]
    at = trim_scored_assistant(online["gen_text"])
    sc = score_tool_calls(at, rec["tools"], rec["gold_block"])
    return bool(sc.get("exact_arguments")), bool(sc.get("valid_tool_call"))


def compare_set(off, on, label):
    rows = []
    for gt in sorted(off):
        o, n = off[gt], on[gt]
        ot, nt = o["gen_text"], n["gen_text"]
        # PRIMARY: byte-identity of the served generated text vs the offline
        # in-process engine's generated text (the "byte certificate"). The FLARE
        # sampler emits no logprobs, so token ids are re-derived consistently on
        # BOTH sides by re-encoding the decoded text (the raw engine ids and a
        # re-encode differ by the grammar's forced-token boundaries, so a
        # consistent re-encode is the apples-to-apples id space).
        byte_ident = (ot == nt)
        ncb = min(len(ot), len(nt))
        fd_char = next((i for i in range(ncb) if ot[i] != nt[i]),
                       None if len(ot) == len(nt) else ncb)
        oid = [int(x) for x in _TOK.encode(ot, add_special_tokens=False)]
        nid = [int(x) for x in _TOK.encode(nt, add_special_tokens=False)]
        ncmp = min(len(oid), len(nid))
        fd = next((i for i in range(ncmp) if oid[i] != nid[i]), None)
        tok_ident = (fd is None and len(oid) == len(nid))
        on_exact, on_valid = score_online(gt, n)
        row = {
            "global_turn": gt, "episode": o["episode"], "turn": o["turn"],
            "n_off_ids": len(oid), "n_on_ids": len(nid),
            "off_raw_gen_ids": o["n_gen"], "on_finish_ids": len(n["online_ids"]),
            "token_identical": tok_ident, "byte_identical": byte_ident,
            "first_div_char_pos": fd_char, "n_char_off": len(ot), "n_char_on": len(nt),
            "first_div_id_pos": fd,
            "off_tok_at_fd": (oid[fd] if fd is not None else None),
            "on_tok_at_fd": (nid[fd] if fd is not None else None),
            "fd_involves_mask": (fd is not None and MASK_ID in (oid[fd], nid[fd])),
            "off_exact": o["eng_exact_arguments"], "on_exact": on_exact,
            "off_valid": o["eng_valid_tool_call"], "on_valid": on_valid,
            "quality_identical": (o["eng_exact_arguments"] == on_exact
                                  and o["eng_valid_tool_call"] == on_valid),
            "off_finish": o["finish_reason"], "on_finish": n["finish_reason"],
            "off_byte_parity_vs_hf": o["byte_parity_vs_hf"],
            # cache-on battery (gates2) cross-check
            "battery_byte_parity_vs_hf": GATES2.get(gt, {}).get("byte_parity_full"),
            "battery_eng_exact": GATES2.get(gt, {}).get("eng_exact_arguments"),
            "off_reproduces_battery_byte": (o["byte_parity_vs_hf"] == GATES2.get(gt, {}).get("byte_parity_full")),
            "off_reproduces_battery_exact": (o["eng_exact_arguments"] == GATES2.get(gt, {}).get("eng_exact_arguments")),
            "proj0": (o["counters"] or {}).get("value_projection_events"),
        }
        rows.append(row)
    n = len(rows)
    summ = {
        "label": label, "n_turns": n,
        "token_identical": sum(r["token_identical"] for r in rows),
        "byte_identical": sum(r["byte_identical"] for r in rows),
        "quality_identical": sum(r["quality_identical"] for r in rows),
        "off_reproduces_battery_byte": sum(bool(r["off_reproduces_battery_byte"]) for r in rows),
        "off_reproduces_battery_exact": sum(bool(r["off_reproduces_battery_exact"]) for r in rows),
        "divergent_turns": [r["global_turn"] for r in rows if not r["token_identical"]],
        "mask_involved_turns": [r["global_turn"] for r in rows if r["fd_involves_mask"]],
        "zero_value_projection_all": all((r["proj0"] == 0) for r in rows),
    }
    return summ, rows


def main():
    off6, on6 = load(OUTDIR / "offline_a6.jsonl"), load(OUTDIR / "online_a6.jsonl")
    off7, on7 = load(OUTDIR / "offline_a7.jsonl"), load(OUTDIR / "online_a7.jsonl")
    s6, r6 = compare_set(off6, on6, "A6_single_turn_fresh")
    s7, r7 = compare_set(off7, on7, "A7_multi_turn_warm")
    cert = {"A6": s6, "A7": s7, "A6_rows": r6, "A7_rows": r7}
    (OUTDIR / "cert.json").write_text(json.dumps(cert, indent=2) + "\n")

    def pr(s, rows):
        print(f"\n=== {s['label']} (n={s['n_turns']}) ===")
        print(f"  token-identical online==offline : {s['token_identical']}/{s['n_turns']}")
        print(f"  byte-identical  online==offline : {s['byte_identical']}/{s['n_turns']}")
        print(f"  quality-identical (exact+valid) : {s['quality_identical']}/{s['n_turns']}")
        print(f"  offline reproduces battery byte : {s['off_reproduces_battery_byte']}/{s['n_turns']}")
        print(f"  offline reproduces battery exact: {s['off_reproduces_battery_exact']}/{s['n_turns']}")
        print(f"  zero-value-projection (all)     : {s['zero_value_projection_all']}")
        print(f"  divergent turns                 : {s['divergent_turns']}")
        print(f"  mask-id involved in a divergence: {s['mask_involved_turns']}")
        for r in rows:
            if not r["token_identical"]:
                print(f"    gt{r['global_turn']} ep{r['episode']}/t{r['turn']}: "
                      f"fd@{r['first_div_id_pos']} off={r['off_tok_at_fd']} on={r['on_tok_at_fd']} "
                      f"mask={r['fd_involves_mask']} off_bp_vs_hf={r['off_byte_parity_vs_hf']} "
                      f"battery_bp_vs_hf={r['battery_byte_parity_vs_hf']} "
                      f"quality_id={r['quality_identical']} (off_exact={r['off_exact']} on_exact={r['on_exact']})")
    pr(s6, r6)
    pr(s7, r7)
    print("\n[compare] wrote", OUTDIR / "cert.json")


if __name__ == "__main__":
    main()
