#!/usr/bin/env python3
"""Assemble the per-capability conversion-tax prompt sets (#28), classes B & C.

Output format mirrors runs/l1_census/gsm8k_prompts_clean.json (class A) EXACTLY so
the same three harnesses (stock-AR offline LLM, merged-AR offline LLM, engine
hybrid_clean) consume all three classes with no code change:

  fields: idx, prompt_ids, prompt_len, block_size(32), mask_id(248077),
          grammar_topk(256), stop_token_ids([248044,248045,248046,248059])
  + per-class scoring metadata (gold/tests/check).

Prompt scaffold matches class A: bare Qwen chat turns (<|im_start|>role ...
<|im_end|>), NO system prompt, thinking-off (assistant content is emitted
directly). Prompt token ids are built with the stock Qwen3.5-9B tokenizer; the
prompt vocab is shared by all three systems (the diffusion mask id 248077 only
appears in the generation canvas, never the prompt), so one id sequence serves
all three — identical to how class A was built and consumed.

B (CODE): MBPP-sanitized (google-research-datasets/mbpp, local HF cache, offline).
  Few-shot = the dataset's designated `prompt` split (task_ids 2,3,4). Eval = the
  first 25 `test`-split problems by ascending task_id. Deterministic.
C (INSTRUCTION): 25 verifiable-constraint prompts constructed here (IFEval-style),
  each with a single deterministic machine check (see runs/conversion_tax/scoring.py).
"""
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("/home/mark/qwen_diffusion")
OUT = ROOT / "runs/conversion_tax"
STOCK = "/home/mark/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
STOP_IDS = [248044, 248045, 248046, 248059]  # <|endoftext|> <|im_start|> <|im_end|> </tool_call>
BLOCK, MASK, GTOPK = 32, 248077, 256
NCODE, NINSTR = 25, 25

tk = AutoTokenizer.from_pretrained(STOCK, trust_remote_code=True)


def ids(s):
    return tk(s, add_special_tokens=False)["input_ids"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- B: CODE (MBPP) --
def code_turn_user(prompt, test_list):
    tests = "\n".join(test_list)
    return (f"<|im_start|>user\nWrite a Python function for the task below. "
            f"Respond with ONLY the complete function inside one ```python code block.\n"
            f"Task: {prompt}\nYour function must pass these tests:\n{tests}<|im_end|>\n")


def code_turn_assistant(code):
    return f"<|im_start|>assistant\n```python\n{code.strip()}\n```<|im_end|>\n"


def build_code():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    fewshot = sorted(ds["prompt"], key=lambda e: e["task_id"])[:3]
    evalset = sorted(ds["test"], key=lambda e: e["task_id"])[:NCODE]

    preamble = ""
    for e in fewshot:
        preamble += code_turn_user(e["prompt"], e["test_list"]) + code_turn_assistant(e["code"])

    recs = []
    for i, e in enumerate(evalset):
        prompt_str = preamble + code_turn_user(e["prompt"], e["test_list"]) + "<|im_start|>assistant\n"
        pid = ids(prompt_str)
        recs.append({
            "idx": i, "task_id": e["task_id"],
            "prompt": e["prompt"], "test_list": e["test_list"],
            "test_imports": e.get("test_imports", []),
            "prompt_ids": pid, "prompt_len": len(pid),
            "block_size": BLOCK, "mask_id": MASK, "grammar_topk": GTOPK,
            "stop_token_ids": STOP_IDS,
        })
    fewshot_ids = [e["task_id"] for e in fewshot]
    eval_ids = [e["task_id"] for e in evalset]
    return recs, fewshot_ids, eval_ids


# -------------------------------------------------- C: INSTRUCTION (built here) --
INSTR = [
    ("Respond with a sentence that is exactly five words long. Output only the sentence.",
     {"type": "word_count_eq", "n": 5}),
    ("Write the word hello in all capital letters. Output only that one word.",
     {"type": "exact_match", "target": "HELLO"}),
    ("Write a single sentence that uses the word banana exactly three times.",
     {"type": "keyword_count_eq", "word": "banana", "n": 3}),
    ("Write one sentence about the sun that does not contain the letter e. Output only the sentence.",
     {"type": "no_letter", "letter": "e"}),
    ("Write two sentences about winter. Your response must end with exactly: The end.",
     {"type": "ends_with", "phrase": "The end."}),
    ("Respond with a JSON object that has exactly two keys, \"name\" and \"age\". Output only the JSON.",
     {"type": "json_keys", "keys": ["name", "age"]}),
    ("List exactly three colors, one per line. Begin each line with the two characters '- '.",
     {"type": "line_prefix", "n": 3, "prefix": "- "}),
    ("Write one sentence that mentions the sun, the moon, and the stars.",
     {"type": "contains_all", "words": ["sun", "moon", "stars"]}),
    ("In fewer than ten words, describe the color of grass.",
     {"type": "word_count_lt", "n": 10}),
    ("Begin your response with the exact word Answer: and then state the capital of France.",
     {"type": "starts_with", "phrase": "Answer:"}),
    ("Write exactly two sentences about your favorite hobby.",
     {"type": "sentence_count_eq", "n": 2}),
    ("Explain in one sentence how plants make food. You must use the word photosynthesis.",
     {"type": "keyword_present", "word": "photosynthesis"}),
    ("Write the phrase i love programming in all lowercase letters. Output only that phrase.",
     {"type": "exact_match", "target": "i love programming"}),
    ("Output only the number 42 and nothing else.",
     {"type": "exact_match", "target": "42"}),
    ("Write one sentence describing a city. Do not use any commas.",
     {"type": "no_comma"}),
    ("List exactly three one-word animal names separated by single spaces. Output only the three words.",
     {"type": "word_count_eq", "n": 3}),
    ("Write the word go four times, separated by single spaces. Output only those words.",
     {"type": "exact_match", "target": "go go go go"}),
    ("Ask a single question about space. Your response must end with a question mark.",
     {"type": "ends_with_char", "char": "?"}),
    ("Write one sentence that contains at least one digit.",
     {"type": "contains_digit"}),
    ("Provide a list of exactly four planets, one per line. Begin each line with the two characters '* '.",
     {"type": "line_prefix", "n": 4, "prefix": "* "}),
    ("Write one sentence about a dog without using the word the.",
     {"type": "no_word", "word": "the"}),
    ("Respond with exactly: OK",
     {"type": "exact_match", "target": "OK"}),
    ("Write a paragraph about the ocean that is at least fifty words long.",
     {"type": "word_count_ge", "n": 50}),
    ("Start your response with the word First and end it with the word Last.",
     {"type": "start_and_end", "start": "First", "end": "Last"}),
    ("List exactly three programming languages, one per line. Begin each line with 'Language: '.",
     {"type": "line_prefix", "n": 3, "prefix": "Language: "}),
]


def build_instr():
    assert len(INSTR) == NINSTR, len(INSTR)
    recs = []
    for i, (instr, check) in enumerate(INSTR):
        # thinking-OFF scaffold (Qwen3.5 enable_thinking=False): zero-shot prompts
        # otherwise open a <think> block and ramble to the length cap. The empty
        # think prefill makes all 3 systems emit a direct, bounded answer.
        prompt_str = (f"<|im_start|>user\n{instr}<|im_end|>\n"
                      f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        pid = ids(prompt_str)
        recs.append({
            "idx": i, "instruction": instr, "check": check,
            "prompt_ids": pid, "prompt_len": len(pid),
            "block_size": BLOCK, "mask_id": MASK, "grammar_topk": GTOPK,
            "stop_token_ids": STOP_IDS,
        })
    return recs


def main():
    code_recs, fs_ids, ev_ids = build_code()
    instr_recs = build_instr()
    cpath = OUT / "code_prompts.json"
    ipath = OUT / "instr_prompts.json"
    cpath.write_text(json.dumps(code_recs))
    ipath.write_text(json.dumps(instr_recs))

    manifest = {
        "class_B_code": {
            "source": "google-research-datasets/mbpp config=sanitized (local HF cache, offline)",
            "cache": "/home/mark/.cache/huggingface/datasets/google-research-datasets___mbpp/sanitized/0.0.0/4bb6404fdc6cacfda99d4ac4205087b89d32030c",
            "fewshot_split": "prompt", "fewshot_task_ids": fs_ids,
            "eval_split": "test", "eval_task_ids": ev_ids, "n": len(code_recs),
            "prompt_shots": 3, "scaffold": "bare Qwen chat turns, no system prompt, thinking-off",
            "scoring": "first ```python fence exec'd vs test_imports+test_list asserts, 5s subprocess timeout; pass = returncode 0",
            "prompt_len_min_max": [min(r["prompt_len"] for r in code_recs), max(r["prompt_len"] for r in code_recs)],
            "sha256": sha(cpath),
        },
        "class_C_instruction": {
            "source": "constructed locally (IFEval-style verifiable constraints), see build_sets.py::INSTR",
            "n": len(instr_recs), "shots": 0,
            "scoring": "single deterministic machine check per item (scoring.py::score_instruction) on stripped completion",
            "check_types": sorted({r["check"]["type"] for r in instr_recs}),
            "prompt_len_min_max": [min(r["prompt_len"] for r in instr_recs), max(r["prompt_len"] for r in instr_recs)],
            "sha256": sha(ipath),
        },
        "shared": {
            "tokenizer": "stock Qwen3.5-9B c202236 (prompt vocab shared by all 3 systems)",
            "block_size": BLOCK, "mask_id": MASK, "grammar_topk": GTOPK,
            "stop_token_ids": STOP_IDS,
            "stop_token_strs": [tk.decode([t]) for t in STOP_IDS],
        },
    }
    (OUT / "prompt_sets_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWROTE {cpath} ({len(code_recs)} recs) and {ipath} ({len(instr_recs)} recs)")


if __name__ == "__main__":
    main()
