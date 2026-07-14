#!/usr/bin/env python3
"""Diff gate-ON vs gate-OFF committed streams (ideal oracle) across cl/align on a
verbatim copy with the '00:00' repeat. Divergence = real seam bug. Uses the
UPSTREAM T._drive for both arms (identical EOS handling) so tail artifacts cancel."""
import os, sys
os.environ["VLLM_FLARE_BIDIR_PROBE"] = "1"
sys.path.insert(0, "/home/mark/shared/vllm_p2_pr42406")
sys.path.insert(0, "/home/mark/shared/vllm_p2_pr42406/tests/v1/sample")
import numpy as np, torch
from types import SimpleNamespace
import test_w1b_engine_seam as T
VOCAB = T.VOCAB_SIZE; EOS = T.EOS_ID


def drive(target, cl, align_off, seed_prompt=None):
    gate_on = seed_prompt is not None
    sampler, ds = T._build_sampler(cl=cl, gate_on=gate_on, max_num_reqs=2)
    slot = 0
    sampler.add_request(slot, prompt_len=0, sampling_params=T._free_text_params())
    sampler._hc_align_off[slot] = align_off
    decoder = sampler._hc_decoders[slot]
    if seed_prompt is not None:
        sampler._hc_prompt_ids[slot] = [ord(c) for c in seed_prompt]
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([slot], dtype=torch.int64))
    ds_slots = torch.tensor([slot], dtype=torch.int64)
    decode_indices_np = np.array([slot]); decode_slots_np = np.array([slot])
    sampler._hc_block_base[slot] = 0
    decoder.bulk_forced_prefix(block_limit=sampler._hc_block_target(slot, 0))
    ds.is_encoder_phase[slot] = sampler._hc_set_next_phase(slot)
    emitted = []; guard = 0
    while guard < 20000 and ((not decoder.finished) or bool(ds.is_encoder_phase[slot])):
        guard += 1
        committing = bool(ds.is_encoder_phase[slot])
        vlen = int(sampler._hc_draft_len.get(slot, 0))
        if vlen <= 0:
            break
        shifted = torch.zeros(1, cl, VOCAB); block_logits = torch.zeros(1, cl, VOCAB)
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


CONTENT = ("def parse_iso8601(value):\n"
           "    if value.endswith('Z'):\n"
           "        value = value[:-1] + '+00:00'\n"
           "    return value")
print("len(CONTENT)=", len(CONTENT))
hits = []
for cl in (8, 16, 32):
    for align in range(cl):
        eoff, doff, _ = drive(CONTENT, cl, align, seed_prompt=None)
        eon, don, son = drive(CONTENT, cl, align, seed_prompt=CONTENT)
        co, cn = dec(doff.committed), dec(don.committed)
        if co != cn:
            i = next((k for k in range(min(len(co), len(cn))) if co[k] != cn[k]),
                     min(len(co), len(cn)))
            hits.append((cl, align))
            print(f"DIVERGE cl={cl} align={align} off_len={len(co)} on_len={len(cn)} "
                  f"spans={son.w1_spans_committed} rej={son.w1_rejects}")
            print(f"   @{i} OFF={co[max(0,i-8):i+10]!r}")
            print(f"   @{i} ON ={cn[max(0,i-8):i+10]!r}")
print("diverging configs:", len(hits))
