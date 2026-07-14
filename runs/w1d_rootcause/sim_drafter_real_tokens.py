#!/usr/bin/env python3
"""Simulate the W1 drafter on the REAL Qwen token ids for idx5. Seed the source
with the prompt; grow the emitted query with the true generated content; at each
denoise step call propose(remaining) with the real block-cap; under the
full-reveal LEAK, model the verify as ACCEPTING whatever byte-copy the drafter
proposes. Check whether the committed stream drops 00' -> reproduces idx5."""
import os, sys
os.environ["FASTDLLM_W1_DRAFT_VERIFY"] = "1"
sys.path.insert(0, "/home/mark/shared/vllm_p2_pr42406")
from transformers import AutoTokenizer
from vllm.v1.sample.w1_draft_verify import W1DraftVerifyController

TOK = AutoTokenizer.from_pretrained(
    "/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-mswe2-S-twinK1-vllm-bf16")

PROMPT = ("Create the file lib/parse.py with exactly this content, using write_file:\n\n"
          "```\n"
          "def parse_iso8601(value: str) -> datetime:\n"
          "    if value.endswith('Z'):\n"
          "        value = value[:-1] + '+00:00'\n"
          "    return datetime.fromisoformat(value)\n"
          "```\n")
CONTENT = ("def parse_iso8601(value: str) -> datetime:\n"
           "    if value.endswith('Z'):\n"
           "        value = value[:-1] + '+00:00'\n"
           "    return datetime.fromisoformat(value)")
prompt_ids = TOK.encode(PROMPT)
gen_ids = TOK.encode(CONTENT)   # the true generated value stream (approx: content only)

BLOCK = 32
# prompt_len for the served request was 513 (from usage). align_off = 513 % 32.
ALIGN_OFF = 513 % BLOCK
print(f"align_off={ALIGN_OFF} block={BLOCK} n_gen={len(gen_ids)}")

ctrl = W1DraftVerifyController()
ctrl.seed_context(prompt_ids)

committed = []            # what actually lands (leak-accept model)
base = 0                  # generated offset of current block start
pos = 0                   # index into TRUE gen stream we are matching against
# We simulate: at each step, if drafter proposes a span, LEAK-accept the whole
# span verbatim (byte-copy) and advance committed by it; else K=1 commit gen[pos].
# We do NOT force draft==truth: we accept whatever the drafter mined.
step = 0
trace = []
while len(committed) < len(gen_ids) and step < 2000:
    step += 1
    # block cap: tokens left in current 32-abs-aligned block
    rem_in_block = BLOCK - ((ALIGN_OFF + len(committed)) % BLOCK)
    dec = ctrl.propose(rem_in_block)
    if dec is not None and dec.committed and len(dec.committed) >= 2:
        span = [int(t) for t in dec.committed]
        # LEAK: accept the mined span verbatim
        true_slice = gen_ids[len(committed):len(committed)+len(span)]
        diverges = span != true_slice
        trace.append((len(committed), "SPAN", len(span), diverges,
                      TOK.decode(span)))
        committed.extend(span)
        ctrl.observe(span)
    else:
        # K=1: commit the true next token
        if len(committed) < len(gen_ids):
            t = gen_ids[len(committed)]
            committed.append(t)
            ctrl.observe([t])
            trace.append((len(committed)-1, "K1", 1, False, TOK.decode([t])))

got = TOK.decode(committed)
print("COMMITTED == TRUE:", got == CONTENT)
print("committed len:", len(committed), "true len:", len(gen_ids))
print("committed text:", repr(got))
print()
print("--- spans proposed (and whether they diverge from truth) ---")
for off, kind, n, div, txt in trace:
    if kind == "SPAN":
        flag = "  <<< DIVERGES" if div else ""
        print(f"  @{off:3d} {kind} len={n} {txt!r}{flag}")
