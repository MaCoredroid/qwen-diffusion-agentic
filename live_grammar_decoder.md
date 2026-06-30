# Live Grammar-Constrained Tool-Call Decoder (banked asset)

**What it is:** an **inference-time, label-free constrained decoder** — NOT a model head, NOT trained weights.
It is a JSON-schema/grammar finite-state machine that runs *inside* the block-diffusion denoising loop and masks
logits so the model can only commit tokens toward a valid tool-call. No retraining; drops onto any checkpoint.

Code: `scripts/eval_fastdllm_toolcall_cases.py` (committed `eb117b4`). Key pieces:
- `apply_live_tool_json_grammar(...)` — per-denoise-step: builds the allowed-token set from the current prefix +
  available tool schema, sets disallowed-token logits to −inf, then samples (top-k) from what remains.
- `tool_call_mode_force_mask(...)` / structural forcing — forces the exact structural literals
  (`<tool_call>`, `\n<function=NAME>\n`, `</function>`, `</tool_call>`) and JSON delimiters.
- `live_tool_json_top_token(...)` — picks the top grammar-valid token; `live_unsafe` counts any forced
  unconstrained fallback (a clean/promotable run requires `live_unsafe=0`).
- **Left-to-right commit during JSON spans:** grammar legality is prefix-dependent, so inside a tool-call the
  decoder commits sequentially (a hybrid — diffusion for the natural-language spans, constrained-AR for the JSON).

**What the grammar constrains (label-free):** the tool-call **structure** (tags/delimiters), the **name** ∈
available tool names, the **argument keys** ∈ the selected tool's schema, and each **value's TYPE** (string /
number / boolean / array / object / enum) per schema. It does NOT supply value *content* — the model picks the
value; the grammar only forbids structurally-invalid tokens. Uses only model output + tool schema + prompt
(`--strip-gold-for-generation`); never gold. Proven label-free: produces 8/8 valid calls with gold **entirely
absent** from the case objects.

## Result (banked)
B@1000 (Hermes-trained student, native eval = TRANSFER), strict (`live_unsafe=0`), label-free, GPU-bound 97% util:
- **exact-args 19/28 (68%)**, **valid JSON 28/28 (100%)**, exact-seq 25/28.
- vs raw 0/28, vs best post-hoc projection 9/28 → the live decoder **>2×** the post-hoc number and guarantees
  valid JSON. Residual 9/28 = value-CONTENT errors (≈ the 70–87% lenient ceiling), not structure.
- This is the project's CONSTRAINED-lane SOTA. A native-TRAINED student × this decoder is the pending finale
  (raw native should lift from B@1000's 0/8).

## Usage
`scripts/eval_fastdllm_toolcall_cases.py ... --full-context-sampling --live-tool-native-grammar
--live-tool-json-topk 1024 --force-tool-call-prefix --strip-gold-for-generation` (block 32, native eval slices
under `data/toolcall_eval_native/`). `topk 1024` + the EOS-after-complete-call fix drive `live_unsafe→0`.

## Format compatibility
Currently targets the **Qwen NATIVE function-call format**:
`<tool_call>\n<function=TOOL_NAME>\n{JSON args}\n</function>\n</tool_call>` (per the native-format standing rule).
- The **structural/tag layer is Qwen-native-specific** (the `<function=...>` body inside the `<tool_call>` wrapper).
- The **value-grounding core is general** (JSON-schema constraint on names/keys/value-types) — portable to other
  tool-call formats (e.g. Hermes-only `<tool_call>{...}`) by swapping the wrapper FSM. So: Qwen-native today,
  generalizable.
