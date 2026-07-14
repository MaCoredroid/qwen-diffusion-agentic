#!/usr/bin/env python3
"""RUNG W-2 ideal-oracle repro: gate-ON (causal-verify + prefix-commit + fixed
block-width + boundary-trim) vs gate-OFF K=1 committed stream, across cl/align.

Divergence = a seam bug in the byte-faithful redesign. Extends the W-1d
repro_seam_ideal_oracle.py: same real _hybrid_clean_step drive, plus a
DIVERGENT-SOURCE case that forces prefix-commit (the causal read matches only a
prefix of the drafted over-copy) and asserts the emitted stream still equals the
K=1 stream byte-for-byte. Writes results.json."""
import json
import os
import sys

os.environ["VLLM_FLARE_BIDIR_PROBE"] = "1"
os.environ["FASTDLLM_W1_DRAFT_VERIFY"] = "1"
sys.path.insert(0, "/home/mark/shared/vllm_p2_pr42406")
sys.path.insert(0, "/home/mark/shared/vllm_p2_pr42406/tests/v1/sample")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import test_w1b_engine_seam as T  # noqa: E402

VOCAB = T.VOCAB_SIZE
EOS = T.EOS_ID


def drive(target, cl, align_off, seed_prompt=None):
    gate_on = seed_prompt is not None
    sampler, ds = T._build_sampler(cl=cl, gate_on=gate_on, max_num_reqs=2)
    slot = 0
    sampler.add_request(slot, prompt_len=0, sampling_params=T._free_text_params())
    sampler._hc_align_off[slot] = align_off
    decoder = sampler._hc_decoders[slot]
    if seed_prompt is not None:
        sampler._hc_prompt_ids[slot] = [ord(c) for c in seed_prompt]
    input_batch = SimpleNamespace(
        idx_mapping=torch.tensor([slot], dtype=torch.int64)
    )
    ds_slots = torch.tensor([slot], dtype=torch.int64)
    decode_indices_np = np.array([slot])
    decode_slots_np = np.array([slot])
    sampler._hc_block_base[slot] = 0
    decoder.bulk_forced_prefix(block_limit=sampler._hc_block_target(slot, 0))
    ds.is_encoder_phase[slot] = sampler._hc_set_next_phase(slot)
    emitted = []
    guard = 0
    while guard < 20000 and (
        (not decoder.finished) or bool(ds.is_encoder_phase[slot])
    ):
        guard += 1
        committing = bool(ds.is_encoder_phase[slot])
        vlen = int(sampler._hc_draft_len.get(slot, 0))
        if vlen <= 0:
            break
        shifted = torch.zeros(1, cl, VOCAB)
        block_logits = torch.zeros(1, cl, VOCAB)
        if committing:
            block_logits[0, vlen - 1, 0] = 1.0
        else:
            k0 = len(decoder.committed)
            ver = sampler._hc_verify.get(slot)
            if ver is not None:
                tl, draft = ver
                for j in range(len(draft)):
                    idx = k0 + j
                    ch = ord(target[idx]) if idx < len(target) else EOS
                    shifted[0, tl + j, ch] = 100.0
            else:
                ch = ord(target[k0]) if k0 < len(target) else EOS
                shifted[0, vlen - 1, ch] = 100.0
        sampler._hybrid_clean_step(
            shifted, block_logits, ds_slots, ds_slots,
            decode_indices_np, decode_slots_np, np.array([vlen]),
            torch.tensor([committing]), 1, input_batch)
        if committing:
            n = int(sampler._num_sampled[slot])
            emitted.extend(int(x) for x in sampler._sampled[slot, :n].tolist())
    return emitted, decoder, sampler


def dec(ids):
    return "".join(chr(int(t)) for t in ids)


# --- Case A: verbatim-copy divergence sweep (base faithfulness) ---
CONTENT = ("def parse_iso8601(value):\n"
           "    if value.endswith('Z'):\n"
           "        value = value[:-1] + '+00:00'\n"
           "    return value")
diverging = []
for cl in (8, 16, 32):
    for align in range(cl):
        eoff, doff, _ = drive(CONTENT, cl, align, seed_prompt=None)
        eon, don, son = drive(CONTENT, cl, align, seed_prompt=CONTENT)
        if dec(doff.committed) != dec(don.committed):
            diverging.append({"cl": cl, "align": align})

# --- Case B: divergent-source PREFIX-COMMIT parity (over-copy is trimmed to the
# matching content prefix; the rest decodes K=1; stream must match K=1). ---
# seed copies "...contentX..." but the true generation continues "...contentY".
targetB = "the quick brown fox jumps over" + " ZZZ"
seedB = "the quick brown fox jumps over" + " QQQ then more copy text here"
prefix_cases = []
for cl in (8, 16):
    eoffB, doffB, _ = drive(targetB, cl, 0, seed_prompt=None)
    eonB, donB, sonB = drive(targetB, cl, 0, seed_prompt=seedB)
    prefix_cases.append({
        "cl": cl,
        "byte_parity": eonB == eoffB,
        "off_stream_matches_target": dec(doffB.committed).startswith(targetB),
        "spans_committed": int(sonB.w1_spans_committed),
        "tokens_committed": int(sonB.w1_tokens_committed),
        "rejects": int(sonB.w1_rejects),
        "verify_forwards": int(sonB.w1_verify_forwards),
        "assert_rejects": int(sonB.w1_assert_rejects),
    })

results = {
    "case_A_divergence_sweep": {
        "configs_tested": 8 + 16 + 32,
        "diverging_configs": len(diverging),
        "diverging": diverging,
    },
    "case_B_prefix_commit_parity": prefix_cases,
    "all_byte_parity": (
        len(diverging) == 0
        and all(c["byte_parity"] for c in prefix_cases)
        and all(c["assert_rejects"] == 0 for c in prefix_cases)
    ),
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_repro.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("PASS" if results["all_byte_parity"] else "FAIL")
