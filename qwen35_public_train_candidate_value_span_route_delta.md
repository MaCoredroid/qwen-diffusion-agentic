# Qwen3.5 Checkpoint-5 Route Delta

Purpose: compare value-span checkpoint-5 against the current split-route target at row level.
This is a diagnostic report, not a training manifest; eval and heldout rows must not be promoted into train data.

## Summary

| Slice | Current protected | Candidate protected | Protected arg delta | Candidate raw/input | Decision signal |
| --- | ---: | ---: | ---: | ---: | --- |
| public one-call | `8/8, 8/8` | `8/8, 8/8` | `+0` | `3/8, 2/8` | tie |
| teacher-train one-call | `11/12, 6/12` | `10/12, 6/12` | `+0` | `2/12, 2/12` | tie |
| teacher-heldout one-call | `8/8, 6/8` | `7/8, 5/8` | `-1` | `1/8, 0/8` | candidate regresses protected args |
| public multi-call planner | `11/12, 10/12` | `11/12, 10/12` | `+0` | `8/12, 5/12` | tie |
| synthetic text tool-result | `10/10, 9/10` | `10/10, 9/10` | `+0` | `5/10, 3/10` | tie |
| OpenAI-style tool-result | `10/10, 9/10` | `10/10, 8/10` | `-1` | `7/10, 7/10` | candidate regresses protected args |

Each metric cell is `exact sequence, exact arguments`.
`raw/input` means the row's unprotected metric field; for chained post-processing lanes this can be the chain input rather than original generation.

## Row-Level Changes

### public one-call

- protected sequence improved: `0`
- protected sequence regressed: `0`
- protected arguments improved: `0`
- protected arguments regressed: `0`
- raw arguments improved: `0`
- raw arguments regressed: `1`

### teacher-train one-call

- protected sequence improved: `0`
- protected sequence regressed: `1`
- protected arguments improved: `0`
- protected arguments regressed: `0`
- raw arguments improved: `0`
- raw arguments regressed: `0`
- top candidate failure paths: `simulate_quantum_entanglement:$` x1, `list_qcaas_providers:$.include_software_tools` x1, `createReservation:$.room_preferences` x1, `book_appointment:$` x1, `generate_invoices:$.client_data` x1

### teacher-heldout one-call

- protected sequence improved: `0`
- protected sequence regressed: `1`
- protected arguments improved: `0`
- protected arguments regressed: `1`
- raw arguments improved: `0`
- raw arguments regressed: `1`
- argument regressions: `c82abc25-206c-4776-a1b1-d6fbc5769bce`
- top candidate failure paths: `debug_quantum_circuit:$.error_types[0]` x1, `debug_quantum_circuit:$.error_types[1]` x1, `debug_quantum_circuit:$.error_types[2]` x1, `debug_quantum_circuit:$.error_types[3]` x1, `debug_quantum_circuit:$.error_types[4]` x1

### public multi-call planner

- protected sequence improved: `0`
- protected sequence regressed: `0`
- protected arguments improved: `0`
- protected arguments regressed: `0`
- raw arguments improved: `4`
- raw arguments regressed: `0`
- top candidate failure paths: `activate_voice_command:$.command` x1, `activate_voice_command:$.device_type` x1, `activate_voice_command:$.location` x1, `configure_motion_detectors:$.installation_code` x1

### synthetic text tool-result

- protected sequence improved: `0`
- protected sequence regressed: `0`
- protected arguments improved: `0`
- protected arguments regressed: `0`
- raw arguments improved: `1`
- raw arguments regressed: `2`
- top candidate failure paths: `send_email:$.subject` x1

### OpenAI-style tool-result

- protected sequence improved: `0`
- protected sequence regressed: `0`
- protected arguments improved: `0`
- protected arguments regressed: `1`
- raw arguments improved: `1`
- raw arguments regressed: `0`
- argument regressions: `synthetic-toolresult-00003`
- top candidate failure paths: `send_email:$.body` x1, `send_email:$.subject` x1, `issue_refund:$.amount` x1

## Training Implications

- Do not train on rows from this eval/heldout delta report.
- Mine train-only analogues for the repeated failure classes: missing one-call tool sequence, scalar argument grounding, and OpenAI-style tool-result argument retention.
- Use checkpoint-5 as positive signal for public multi-call constrained/contextual row grounding, not as a promoted route.

## Artifacts

- JSON: `runs/fastdllm_qwen35_9b_candidate_value_span_public_train_from_ckpt275_step10/checkpoint-5/route_delta_vs_current_routed_target.json`
