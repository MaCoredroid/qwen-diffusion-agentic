#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_FRAGMENTS = [
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    '{"name": ',
    '"name"',
    '"arguments"',
    "name",
    "arguments",
    "{",
    "}",
    "[",
    "]",
    ":",
    ",",
    '"',
    "true",
    "false",
    "null",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive structural token IDs for Fast-DLLM tool-call loss weighting."
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init",
        help="Tokenizer path or HF id.",
    )
    parser.add_argument(
        "--fragment",
        action="append",
        default=[],
        help="Extra fragment to tokenize and include. Can be repeated.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional manifest path with fragment-to-token mapping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fragments = list(dict.fromkeys(DEFAULT_FRAGMENTS + args.fragment))
    mapping = {}
    token_ids = set()
    for fragment in fragments:
        ids = tokenizer.encode(fragment, add_special_tokens=False)
        mapping[fragment] = ids
        token_ids.update(ids)

    ordered_ids = sorted(token_ids)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "tokenizer": args.tokenizer,
                    "fragments": mapping,
                    "token_ids": ordered_ids,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(",".join(str(token_id) for token_id in ordered_ids))


if __name__ == "__main__":
    main()
