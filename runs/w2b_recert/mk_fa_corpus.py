#!/usr/bin/env python3
"""Build a 12-case near-dup pointer-slip FA battery (write_file) at temp 0.6.

Each file embeds several NEAR-DUPLICATE blocks that share a long copyable prefix
and diverge at exactly one token (a return value / constant / attr name). The
copy drafter mines both occurrences, so at the divergence the guard sees >=2
candidates and must NOT slip to the wrong continuation. The whole target is in
the prompt, so the K=1 (gate-OFF) schedule reproduces it byte-for-byte; a false
accept is then EXACTLY a gate-ON byte-divergence from gate-OFF. Correct
resolution = gate-ON == gate-OFF on all 12 (checked by compare06.py).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "qwen3.5-9b-flare-hybrid-clean"

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write the given content to the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


def near_dup_file(names, consts):
    """A file with len(names) near-dup functions differing only in a constant."""
    lines = []
    for nm, c in zip(names, consts):
        lines.append(f"def {nm}(x, y):")
        lines.append("    total = x + y")
        lines.append("    total = total * 2")
        lines.append(f"    return total + {c}")
        lines.append("")
    return "\n".join(lines)


def near_dup_dict(keys, vals):
    lines = ["CONFIG = {"]
    for k, v in zip(keys, vals):
        lines.append(f'    "{k}": {{')
        lines.append('        "enabled": True,')
        lines.append('        "retries": 3,')
        lines.append(f'        "weight": {v},')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


# 12 cases: 6 near-dup-function files + 6 near-dup-dict files, varied so the
# divergent token lands at different depths.
CASES = []
fn_specs = [
    ("a.py", ["alpha", "beta", "gamma"], [1, 2, 3]),
    ("m.py", ["load", "save", "sync"], [10, 11, 12]),
    ("s.py", ["parse", "build", "merge"], [7, 8, 9]),
    ("r.py", ["read", "write", "flush"], [100, 200, 300]),
    ("k.py", ["push", "pull", "peek"], [5, 6, 4]),
    ("d.py", ["open_", "close_", "reset_"], [21, 22, 23]),
]
dict_specs = [
    ("cfg.py", ["fast", "slow", "auto"], [1, 2, 3]),
    ("t.py", ["cpu", "gpu", "tpu"], [8, 16, 32]),
    ("v.py", ["red", "green", "blue"], [255, 128, 64]),
    ("n.py", ["north", "south", "east"], [0, 90, 180]),
    ("p.py", ["low", "mid", "high"], [1, 5, 9]),
    ("c.py", ["dev", "test", "prod"], [11, 22, 33]),
]
for path, names, consts in fn_specs:
    CASES.append((path, near_dup_file(names, consts)))
for path, keys, vals in dict_specs:
    CASES.append((path, near_dup_dict(keys, vals)))


def build():
    rows = []
    for idx, (path, content) in enumerate(CASES):
        user = (
            f"Create the file {path} with exactly this content, using "
            f"write_file:\n\n```\n{content}\n```"
        )
        rows.append({
            "model": MODEL,
            "messages": [{"role": "user", "content": user}],
            "tools": [WRITE_FILE_TOOL],
            "max_tokens": 400,
            "temperature": 0.0,  # overridden to 0.6 by the driver
            "stream": False,
            "_idx": idx,
            "_rep": 0,
            "_class": "fa_near_dup",
            "_path": path,
            "_want_content": content,
        })
    with open(f"{HERE}/fa_corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gold = {str(i): CASES[i][1] for i in range(len(CASES))}
    json.dump(gold, open(f"{HERE}/fa_gold.json", "w"), indent=1)
    print(f"wrote {len(rows)} FA rows + gold")


if __name__ == "__main__":
    build()
