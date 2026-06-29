#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive token IDs that bound Fast-DLLM tool-call argument spans."
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/mark/qwen_diffusion/models/qwen3.5-9b-fastdllm-init",
        help="Tokenizer path or HF id.",
    )
    parser.add_argument(
        "--start-fragment",
        default="arguments",
        help="Fragment that marks the start of an argument span.",
    )
    parser.add_argument(
        "--end-fragment",
        default="</tool_call>",
        help="Fragment that marks the end of an argument span.",
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
    start_ids = tokenizer.encode(args.start_fragment, add_special_tokens=False)
    end_ids = tokenizer.encode(args.end_fragment, add_special_tokens=False)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "tokenizer": args.tokenizer,
                    "start_fragment": args.start_fragment,
                    "start_token_ids": start_ids,
                    "end_fragment": args.end_fragment,
                    "end_token_ids": end_ids,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        ",".join(str(token_id) for token_id in start_ids)
        + "\t"
        + ",".join(str(token_id) for token_id in end_ids)
    )


if __name__ == "__main__":
    main()
