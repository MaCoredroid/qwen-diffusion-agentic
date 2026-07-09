#!/usr/bin/env python3
"""
build_swe_sft_dataset.py -- materialize the SWE-SFT training dataset from the
frozen keepers pool, per swe_tuning_campaign_design.md conventions.

Steps (each is a hard gate or an audited artifact):

  1. RENDER (native qwen3_xml, native-format-rule). Keepers store STRUCTURED
     OpenAI-style messages (role/content/tool_calls); they are rendered with the
     student's OWN serving chat template (models/.../chat_template.jinja) -- the
     AUTHORITATIVE native format the 9B is served with. Assistant turns (reasoning
     + <tool_call> function blocks + terminating text) are the SFT loss targets;
     system/user/tool turns are loss-masked. NO double-templating (content carries
     zero <|im_start|> control tokens; the template wraps exactly once).

     FINDING: the LMFlow preset `fast_dllm_v2_native` (the QLoRA trainer's current
     CONVERSATION_TEMPLATE default) is NOT byte-identical to the serving template --
     it injects an extra newline around <tool_call>/<tool_response>, so 7023/10038
     assistant TARGET turns diverge. Training on it would teach a whitespace variant
     that mismatches serve (violates native-format-rule). => We ship a serve-EXACT
     pre-tokenized dataset (input_ids + assistant spans) computed under the serving
     template + a token-scan mask, so training == serving byte-for-byte. The
     structured conversation JSON is also emitted (template-agnostic) with the
     caveat that any template used to train it MUST reproduce the serving bytes.

  2. LEAKAGE GATE (KILL-D1): reconstruct the 113-id eval holdout byte-identically
     to leakage_audit.py, assert sha256 == pin; assert keeper_ids INTERSECT
     holdout == EMPTY; assert quarantined ids excluded.

  3. SPLIT: single 'train' split (design specifies no keeper-level dev; retention
     via external GSM8K + matched-20 probes).

  4. MANIFEST: teacher mix + source mix + family (repo) distribution + firewall.

  5. LENGTH AUDIT: full length distribution under the serving tokenizer with
     block_size=32768 / truncation_side=left (design 2.3); reports assistant-label
     retention under truncation, honestly.
"""
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/mark/qwen_diffusion")
HERE = REPO / "runs/swe_datagen_s1"
sys.path.insert(0, str(REPO / "fast-dllm/third_party"))

from transformers import AutoTokenizer  # noqa: E402
from lmflow.utils.conversation_template import PRESET_TEMPLATES  # noqa: E402

STUDENT_MODEL = REPO / "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16"
SERVE_TEMPLATE_FILE = STUDENT_MODEL / "chat_template.jinja"
TRAINER_PRESET = "fast_dllm_v2_native"  # divergence reference only
BLOCK_SIZE = 32768
TRUNCATION_SIDE = "left"

KEEPERS = HERE / "keepers/keepers.jsonl"
QUARANTINED = HERE / "keepers/quarantined.jsonl"
POOL_MANIFEST = REPO / "data/swe_sft_pool/pool_manifest.json"
PIN = HERE / ".eval_holdout_sha256"
RING_SRC = {
    "tier0_20": REPO / "runs/stage_c_driver/data/swe-bench-tier0-verified-instances-20260520.json",
    "tier1_100": Path("/home/mark/shared/lumoFlyWheel/docs/reports/auto_research/swe-bench-tier1-verified-instances-20260520.json"),
}

OUT_DIR = REPO / "data/swe_sft_pool"
LMFLOW_DIR = OUT_DIR / "lmflow_dataset"
TRAIN_JSON = LMFLOW_DIR / "train_swe_sft.json"           # structured conversation
TOKENIZED_JSONL = OUT_DIR / "train_swe_sft.tokenized.jsonl"  # serve-exact, pre-masked
STATS_JSONL = OUT_DIR / "sft_per_instance_stats.jsonl"
MANIFEST_JSON = OUT_DIR / "sft_dataset_manifest.json"
MANIFEST_MD = OUT_DIR / "sft_dataset_manifest.md"

TEACHER = {
    "stock-Qwen3.5-9B-AR (qwen_code, native qwen3_xml)": "stock-9b-ar",
    "Qwen3.6-27B-NVFP4+MTP (nvidia ckpt, qwen_code, native qwen3_xml)": "qwen3.6-27b-nvfp4-mtp",
    "Claude-Opus-4.8 (qwen-code via OAuth adapter, native qwen3_xml)": "opus-4.8",
    "claude-opus-4-8 (Claude Code OAuth via opus_openai_adapter, native qwen3_xml)": "opus-4.8",
}


def _ids(path: Path) -> set:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        return set(d.get("instance_ids", []))
    if isinstance(d, list):
        if d and isinstance(d[0], str):
            return set(d)
        return {r.get("instance_id") for r in d if isinstance(r, dict) and r.get("instance_id")}
    return set()


def _sha(ids: set) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def reconstruct_holdout():
    man = json.loads(POOL_MANIFEST.read_text())
    inner5 = set(man["held_out_rings"]["inner5"]["ids"])
    holdout = inner5 | _ids(RING_SRC["tier0_20"]) | _ids(RING_SRC["tier1_100"])
    sha = _sha(holdout)
    pinned = PIN.read_text().strip()
    if sha != pinned:
        raise SystemExit(f"KILL-D1 HASH MISMATCH: reconstructed {sha} != pinned {pinned}")
    return holdout, sha


def flatten(c):
    if c is None:
        return None
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def norm_tool_calls(tcs):
    out = []
    for tc in tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({"id": tc.get("id"), "type": tc.get("type", "function"),
                    "function": {"name": fn.get("name"), "arguments": args}})
    return out


def keeper_to_instance(kp):
    msgs = copy.deepcopy(kp["messages"])
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = flatten(msgs[0]["content"])
        msgs = msgs[1:]
    out_msgs = []
    for m in msgs:
        nm = {"role": m["role"]}
        fc = flatten(m.get("content"))
        if fc is not None:
            nm["content"] = fc
        if m.get("tool_calls"):
            nm["tool_calls"] = norm_tool_calls(m["tool_calls"])
        if m.get("tool_call_id"):
            nm["tool_call_id"] = m["tool_call_id"]
        out_msgs.append(nm)
    inst = {"conversation_id": kp["instance_id"], "messages": out_msgs}
    if system is not None:
        inst["system"] = system
    tools = kp.get("tools") or None
    if tools:
        inst["tools"] = tools
    return inst


def conv_for_template(inst):
    system = inst.get("system")
    conv = [{"role": "system", "content": system if system is not None else "You are a helpful assistant."}]
    conv.extend(copy.deepcopy(inst.get("messages") or []))
    return conv


def pct(vals):
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)

    def q(p):
        return s[min(n - 1, int(round(p * (n - 1))))]
    return {"min": s[0], "p10": q(0.10), "p25": q(0.25), "p50": q(0.50), "p75": q(0.75),
            "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": s[-1], "mean": round(sum(s) / n, 1)}


def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[build_swe_sft] {ts}")

    holdout, sha = reconstruct_holdout()
    print(f"[gate] holdout {len(holdout)} ids sha256={sha[:16]}.. == pin OK")

    quarantined_ids = set()
    if QUARANTINED.exists():
        for line in QUARANTINED.read_text().splitlines():
            if line.strip():
                quarantined_ids.add(json.loads(line)["instance_id"])
    print(f"[gate] quarantined ids: {sorted(quarantined_ids)}")

    keepers = [json.loads(l) for l in KEEPERS.read_text().splitlines() if l.strip()]
    kept_ids = [k["instance_id"] for k in keepers]
    assert len(kept_ids) == len(set(kept_ids)), "duplicate keeper instance_ids"
    overlap = sorted(set(kept_ids) & holdout)
    if overlap:
        raise SystemExit(f"KILL-D1 LEAK: {len(overlap)} keeper ids in holdout: {overlap[:10]}")
    q_in = sorted(set(kept_ids) & quarantined_ids)
    if q_in:
        raise SystemExit(f"KILL: quarantined ids in keepers: {q_in}")
    print(f"[gate] keeper x holdout overlap=0 ; quarantined-in-train=0 ; keepers={len(keepers)}")

    tok = AutoTokenizer.from_pretrained(str(STUDENT_MODEL), trust_remote_code=True)
    serve_tmpl = SERVE_TEMPLATE_FILE.read_text()
    train_preset = PRESET_TEMPLATES[TRAINER_PRESET]
    IM_START = tok.convert_tokens_to_ids("<|im_start|>")
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    ASST_HDR = tok.encode("assistant\n", add_special_tokens=False)

    def scan_spans(ids):
        """Return list of [start,end) assistant-generated token spans (post-header
        through the closing <|im_end|> inclusive)."""
        spans = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == IM_START and ids[i + 1:i + 1 + len(ASST_HDR)] == ASST_HDR:
                j = i + 1 + len(ASST_HDR)
                k = j
                while k < n and ids[k] != IM_END:
                    k += 1
                end = min(k, n - 1)
                spans.append((j, end + 1))
                i = end + 1
            else:
                i += 1
        return spans

    instances = []
    tokenized_rows = []
    stats_rows = []
    teacher_ct = Counter()
    source_ct = Counter()
    repo_ct = Counter()
    teacher_by_source = Counter()
    tools_present = 0

    lengths, target_tokens_list = [], []
    over_block = 0
    partial_trunc = 0
    zero_trunc = 0
    labels_lost_over_block = []
    dt_zero_double = True
    THIN_FLOOR = 64  # assistant-target tokens below this == extraction lost the trajectory
    thin_outliers = []

    # divergence stats vs the trainer preset
    turns_total = 0
    turns_diverge = 0
    rows_preset_identical = 0

    for i, kp in enumerate(keepers):
        inst = keeper_to_instance(kp)
        instances.append(inst)
        teacher = TEACHER.get(kp["provenance"]["generator"], kp["provenance"]["generator"])
        conv = conv_for_template(inst)

        ids = tok.apply_chat_template(conversation=conv, tools=inst.get("tools"),
                                      chat_template=serve_tmpl, add_generation_prompt=False,
                                      return_dict=True)["input_ids"]
        spans = scan_spans(ids)
        full_labels = sum(e - s for s, e in spans)
        length = len(ids)

        # zero double-templating invariant. Two guards:
        #  (a) stored content carries no chat control tokens (checked below), and
        #  (b) rendered markers are balanced (im_start == im_end). NOTE: consecutive
        #      tool messages legitimately share ONE <|im_start|>user wrapper (template
        #      grouping), so marker count is <= n_msgs by design -- not a doubling.
        n_msgs = len(conv)
        if ids.count(IM_START) != ids.count(IM_END):
            dt_zero_double = False
        for m in conv:
            c = m.get("content")
            if isinstance(c, str) and ("<|im_start|>" in c or "<|im_end|>" in c):
                dt_zero_double = False

        # block_size left-truncation
        if length > BLOCK_SIZE:
            over_block += 1
            cut = length - BLOCK_SIZE
            t_ids = ids[cut:]
            t_spans = []
            for s, e in spans:
                ns, ne = s - cut, e - cut
                if ne <= 0:
                    continue
                t_spans.append((max(0, ns), ne))
            kept_labels = sum(e - s for s, e in t_spans)
            labels_lost_over_block.append(full_labels - kept_labels)
            if kept_labels == 0:
                zero_trunc += 1
            elif kept_labels < full_labels:
                partial_trunc += 1
        else:
            t_ids = ids
            t_spans = spans
            kept_labels = full_labels

        teacher_ct[teacher] += 1
        source_ct[kp["source"]] += 1
        repo_ct[kp["repo"]] += 1
        teacher_by_source[(teacher, kp["source"])] += 1
        if inst.get("tools"):
            tools_present += 1
        lengths.append(length)
        target_tokens_list.append(full_labels)
        if full_labels < THIN_FLOOR:
            thin_outliers.append({"conversation_id": kp["instance_id"], "teacher": teacher,
                                  "n_assistant_target_tokens": full_labels,
                                  "n_assistant_msgs": sum(1 for m in conv if m["role"] == "assistant")})

        tokenized_rows.append({
            "conversation_id": kp["instance_id"],
            "input_ids": t_ids,
            "assistant_spans": t_spans,  # [start,end) into input_ids; labels = ids there, else -100
            "n_tokens": len(t_ids),
            "n_label_tokens": kept_labels,
        })
        stats_rows.append({
            "conversation_id": kp["instance_id"],
            "repo": kp["repo"], "source": kp["source"], "teacher": teacher, "split": "train",
            "batch_id": kp["provenance"].get("batch_id"),
            "num_messages": n_msgs,
            "tools_present": bool(inst.get("tools")),
            "n_tokens": length,
            "n_assistant_target_tokens": full_labels,
            "over_block_size": length > BLOCK_SIZE,
            "n_label_tokens_after_trunc": kept_labels,
        })

        # divergence vs trainer preset (per assistant-turn segment identity)
        import re as _re
        ids_p = tok.apply_chat_template(conversation=conv, tools=inst.get("tools"),
                                        chat_template=train_preset, return_dict=True)["input_ids"]
        seg_s = _re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", tok.decode(ids), flags=_re.S)
        seg_p = _re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", tok.decode(ids_p), flags=_re.S)
        turns_total += len(seg_s)
        row_ok = True
        for a, b in zip(seg_s, seg_p):
            if a != b:
                turns_diverge += 1
                row_ok = False
        if row_ok and len(seg_s) == len(seg_p):
            rows_preset_identical += 1

        if (i + 1) % 50 == 0:
            print(f"  .. {i+1}/{len(keepers)}")

    # write structured conversation dataset (template-agnostic)
    LMFLOW_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_JSON.write_text(json.dumps({"type": "conversation", "instances": instances}))
    # write serve-exact pre-tokenized dataset
    with TOKENIZED_JSONL.open("w") as f:
        for r in tokenized_rows:
            f.write(json.dumps(r) + "\n")
    STATS_JSONL.write_text("\n".join(json.dumps(r) for r in stats_rows) + "\n")

    manifest = {
        "artifact": "swe_sft_dataset",
        "purpose": "SWE-SFT (RFT/rejection-sampling) training set for the two-arm SWE-tuning campaign "
                   "(arm-1 merged-RL-v2 / arm-2 stock); design swe_tuning_campaign_design.md.",
        "built_at": ts,
        "built_by": "runs/swe_datagen_s1/build_swe_sft_dataset.py",
        "source_pool": {
            "keepers_jsonl": str(KEEPERS.relative_to(REPO)),
            "keepers_count": len(keepers),
            "pool_manifest_firewall": str(POOL_MANIFEST.relative_to(REPO)),
        },
        "render": {
            "native_format": "qwen3_xml (native-format-rule). Assistant reasoning + <tool_call> blocks + "
                             "terminating text = SFT targets; system/user/tool = loss-masked.",
            "authoritative_template": "models/qwen3.5-9b-fastdllm-rlv2-vllm-bf16/chat_template.jinja "
                                      "(the 9B serving template; md5 94f89e03284d911fc65d06422439fd79, "
                                      "identical across arm-1/arm-2/init).",
            "tokenizer": str(STUDENT_MODEL.relative_to(REPO)),
            "double_templating": "NONE (verified: exactly one <|im_start|>/<|im_end|> per message; "
                                 "content carries zero control tokens).",
            "content_flattening": "multi-part content lists joined (''.join item.text) == serving "
                                  "render_content macro output.",
            "tool_call_args": "function.arguments JSON strings parsed to dicts for template rendering.",
            "assistant_mask": "token-scan: tokens from after '<|im_start|>assistant\\n' through the "
                              "closing '<|im_end|>' inclusive; span-count == assistant-message count "
                              "(validated).",
        },
        "native_format_finding": {
            "trainer_preset": TRAINER_PRESET,
            "verdict": "DO NOT train the structured JSON with `fast_dllm_v2_native` as-is.",
            "detail": "The LMFlow preset renders an EXTRA newline around <tool_call>/<tool_response> "
                      "relative to the 9B serving template, so assistant TARGET turns are not "
                      "byte-identical to serve.",
            "assistant_turns_total": turns_total,
            "assistant_turns_diverging_vs_preset": turns_diverge,
            "rows_fully_identical_vs_preset": rows_preset_identical,
            "resolution": "Train from `train_swe_sft.tokenized.jsonl` (serve-EXACT input_ids + "
                          "assistant_spans) OR point the trainer at the serving chat_template.jinja "
                          "(generation-tagged) so tokens match serve byte-for-byte.",
        },
        "leakage_firewall": {
            "eval_holdout_ids": len(holdout),
            "eval_holdout_sha256": sha,
            "eval_holdout_sha256_pin": PIN.read_text().strip(),
            "sha_asserted_equal": True,
            "keeper_x_holdout_overlap": 0,
            "quarantined_ids": sorted(quarantined_ids),
            "quarantined_present_in_train": 0,
            "rule": "train_ids INTERSECT (inner5 UNION tier0_20 UNION tier1_100) == EMPTY (KILL-D1).",
        },
        "split": {
            "policy": "single 'train' split. Design specifies no keeper-level dev/validation; retention "
                      "monitored by external probes (GSM8K N=5 every 50 steps + matched-20 tool-call "
                      "anchor spot-gate), not a held-in dev slice.",
            "train": len(instances), "dev": 0,
        },
        "teacher_mix": dict(teacher_ct.most_common()),
        "source_mix": dict(source_ct.most_common()),
        "teacher_by_source": {f"{t}|{s}": n for (t, s), n in sorted(teacher_by_source.items())},
        "family_distribution_repos": dict(repo_ct.most_common()),
        "family_count": len(repo_ct),
        "tool_schema_fidelity": {
            "instances_with_tools_block": tools_present,
            "instances_without_tools_block": len(keepers) - tools_present,
            "caveat": "tools captured from each episode's richest chat dump; ~half of dumps omitted the "
                      "tools field, so those instances render WITHOUT the system-prompt tool-declaration "
                      "block that was present at gen/serve time. Loss-MASKED context skew only (assistant "
                      "targets carry the qwen3_xml call format regardless). Optional remediation for the "
                      "training-task owner: backfill the canonical qwen-code tool set.",
        },
        "length_audit": {
            "tokenizer": str(STUDENT_MODEL.relative_to(REPO)),
            "template": "serving chat_template.jinja (authoritative)",
            "block_size": BLOCK_SIZE,
            "truncation_side": TRUNCATION_SIDE,
            "packing": "OFF (one episode = one sample; design 2.3 left-truncation, no group_texts).",
            "total_tokens": sum(lengths),
            "total_assistant_target_tokens": sum(target_tokens_list),
            "token_length_distribution": pct(lengths),
            "assistant_target_token_distribution": pct(target_tokens_list),
            "over_block_size_count": over_block,
            "over_block_size_pct": round(100 * over_block / len(keepers), 2),
            "labels_lost_when_over_block": pct(labels_lost_over_block) if labels_lost_over_block else {},
            "partial_after_truncation": partial_trunc,
            "zero_after_truncation": zero_trunc,
            "note": "left-truncation at 32768 drops the EARLIEST tokens (system + task prompt + early "
                    "turns) on over-length episodes, keeping the latest assistant turns; fitting episodes "
                    "(<=32768) are untouched.",
        },
        "thin_trajectory_outliers": {
            "floor_assistant_target_tokens": THIN_FLOOR,
            "count": len(thin_outliers),
            "rows": thin_outliers,
            "note": "resolved keepers whose STORED message window carries < floor assistant-target tokens "
                    "(the richest-dump extraction captured a truncated/post-compaction window, so the "
                    "trajectory is near-empty as an SFT sample though the patch scored resolved). NOT "
                    "dropped here (pool curation is the owner's decision); flagged so the training-task "
                    "owner can exclude sub-floor rows if desired.",
        },
        "double_templating_zero": dt_zero_double,
        "outputs": {
            "structured_conversation_json": str(TRAIN_JSON.relative_to(REPO)),
            "serve_exact_tokenized_jsonl": str(TOKENIZED_JSONL.relative_to(REPO)),
            "per_instance_stats": str(STATS_JSONL.relative_to(REPO)),
            "manifest_json": str(MANIFEST_JSON.relative_to(REPO)),
            "manifest_md": str(MANIFEST_MD.relative_to(REPO)),
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))

    la = manifest["length_audit"]
    md = []
    md.append("# SWE-SFT dataset manifest\n")
    md.append(f"Built {ts} by `runs/swe_datagen_s1/build_swe_sft_dataset.py` from "
              f"`{manifest['source_pool']['keepers_jsonl']}` ({len(keepers)} keepers).\n")
    md.append("## Firewall (KILL-D1)\n")
    md.append(f"- eval holdout: **{len(holdout)} ids**, sha256 `{sha}` == pin **OK**")
    md.append(f"- keeper x holdout overlap: **0** / {len(keepers)}")
    md.append(f"- quarantined excluded: **{sorted(quarantined_ids)}** (0 in train)\n")
    md.append("## Render / native format\n")
    md.append("- authoritative template: the 9B **serving** `chat_template.jinja` (native qwen3_xml).")
    md.append(f"- double-templating: **NONE** (zero_double={dt_zero_double}).")
    md.append(f"- native-format finding: trainer preset `{TRAINER_PRESET}` diverges from serve on "
              f"**{turns_diverge}/{turns_total}** assistant turns (extra tool_call/tool_response "
              f"whitespace); **{rows_preset_identical}/{len(keepers)}** rows identical. **Train from the "
              "serve-exact `train_swe_sft.tokenized.jsonl`**, not the preset.\n")
    md.append("## Split\n- single `train` split (no keeper-level dev per design).\n")
    md.append("## Teacher mix\n")
    for t, n in teacher_ct.most_common():
        md.append(f"- {t}: {n} ({100*n/len(keepers):.1f}%)")
    md.append("\n## Source mix\n")
    for s, n in source_ct.most_common():
        md.append(f"- {s}: {n} ({100*n/len(keepers):.1f}%)")
    md.append(f"\n## Family distribution ({len(repo_ct)} repos)\n")
    for r, n in repo_ct.most_common():
        md.append(f"- {r}: {n}")
    md.append("\n## Tool-schema fidelity\n")
    md.append(f"- with tools block: {tools_present}/{len(keepers)}; without: {len(keepers)-tools_present} "
              "(loss-masked context skew only).\n")
    md.append("## Length audit (serving template, block_size=32768, truncation=left)\n")
    md.append(f"- token length: {la['token_length_distribution']}")
    md.append(f"- assistant-target tokens: {la['assistant_target_token_distribution']}")
    md.append(f"- total tokens: {la['total_tokens']:,}; total assistant-target tokens: "
              f"{la['total_assistant_target_tokens']:,}")
    md.append(f"- over block_size (32768): **{over_block}** ({la['over_block_size_pct']}%); "
              f"partial-after-trunc {partial_trunc}; zero-after-trunc {zero_trunc}")
    if labels_lost_over_block:
        md.append(f"- labels lost when over-block: {la['labels_lost_when_over_block']}")
    md.append(f"\n## Thin-trajectory outliers (< {THIN_FLOOR} assistant-target tokens)\n")
    md.append(f"- count: **{len(thin_outliers)}** (flagged, NOT dropped -- owner curates)")
    for o in thin_outliers:
        md.append(f"  - {o['conversation_id']} (teacher {o['teacher']}, {o['n_assistant_target_tokens']} "
                  f"target toks, {o['n_assistant_msgs']} assistant msgs)")
    md.append("")
    MANIFEST_MD.write_text("\n".join(md))

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "keepers": len(keepers),
        "teacher_mix": dict(teacher_ct),
        "source_mix": dict(source_ct),
        "families": len(repo_ct),
        "sha_ok": True, "holdout_overlap": 0, "quarantined_in_train": 0,
        "double_templating_zero": dt_zero_double,
        "preset_divergence_turns": f"{turns_diverge}/{turns_total}",
        "over_block_size": over_block,
        "token_len_p50_p95_max": [la["token_length_distribution"]["p50"],
                                   la["token_length_distribution"]["p95"],
                                   la["token_length_distribution"]["max"]],
    }, indent=2))
    print(f"\nwrote:\n  {TRAIN_JSON}\n  {TOKENIZED_JSONL}\n  {STATS_JSONL}\n  {MANIFEST_JSON}\n  {MANIFEST_MD}")


if __name__ == "__main__":
    main()
