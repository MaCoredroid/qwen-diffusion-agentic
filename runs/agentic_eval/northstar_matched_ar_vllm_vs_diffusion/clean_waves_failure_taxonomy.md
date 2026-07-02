# Clean Waves Failure Taxonomy

Scope: matched-20 strict clean-waves failures after 19a50b7 generated-token audit. Rows are prioritized where diffusion-careful succeeds and strict waves fail.

Strict failures: 60/63 turns. Failure classes: malformed_xml_or_tool_block=17, schema_or_required_arg_failure=26, valid_schema_wrong_argument_values=7, wrong_or_missing_tool_sequence=10.
All selected rows have `projected_value_tokens_exact=0`; the collapse is not residual value projection contamination.

| # | episode | turn | class | careful exact | AR-guided exact | strict tools | gold value shapes | forwards | projected scaffold | first mismatch |
|---:|---|---:|---|---:|---:|---|---|---:|---:|---|
| 1 | heldout_seed_run1clean_0000 | 0 | valid_schema_wrong_argument_values | 1 | 1 | initialize_qubits | {"string": 2} | 32 | 10 | argument_value: initialize_qubits.initial_state gold='00' pred='000' |
| 2 | heldout_seed_run1clean_0001 | 2 | wrong_or_missing_tool_sequence | 1 | 1 | index_documents | {"string": 2} | 136 | 4 | tool_name: gold=setup_case_timeline pred=index_documents |
| 3 | heldout_seed_run1clean_0002 | 0 | valid_schema_wrong_argument_values | 1 | 1 | categorize_records_by_diagnosis | {"string": 1} | 68 | 10 | argument_value: categorize_records_by_diagnosis.records_path gold='/path/to/patient/records' pred='/path/to/patient/records\n\n<tool_' |
| 4 | heldout_seed_run1clean_0002 | 1 | wrong_or_missing_tool_sequence | 1 | 1 | update_records_with_latest_visit | {"string": 1} | 50 | 5 | tool_name: gold=list_upcoming_appointments pred=update_records_with_latest_visit |
| 5 | heldout_seed_run1clean_0002 | 2 | malformed_xml_or_tool_block | 1 | 1 | - | {"string": 2} | 41 | 13 | call_count: gold=1 pred=0 |
| 6 | heldout_seed_run1clean_0003 | 0 | schema_or_required_arg_failure | 1 | 1 | synchronizeRoomAvailability | {"string": 3} | 143 | 8 | argument_value: synchronizeRoomAvailability.automateGuestCheckInOut gold='None' pred='<parameter=hotel_id>\nH1001' |
| 7 | heldout_seed_run1clean_0003 | 1 | malformed_xml_or_tool_block | 1 | 1 | - | {"bool": 2, "string": 1} | 55 | 4 | call_count: gold=1 pred=0 |
| 8 | heldout_seed_run1clean_0003 | 2 | wrong_or_missing_tool_sequence | 1 | 1 | automateGuestCheckInOut | {"string": 3} | 54 | 6 | tool_name: gold=scheduleHousekeeping pred=automateGuestCheckInOut |
| 9 | heldout_seed_run1clean_0004 | 0 | schema_or_required_arg_failure | 1 | 1 | record_project_expense | {"string": 7} | 86 | 8 | argument_value: record_project_expense.currency gold='USD' pred='None' |
| 10 | heldout_seed_run1clean_0004 | 1 | schema_or_required_arg_failure | 1 | 1 | record_project_expense | {"string": 7} | 64 | 8 | argument_value: record_project_expense.amount gold='4500' pred='None' |

Diagnosis: strict waves are audit-clean but fragile. The strict scheduler only projects forced scaffold and refuses to project inside values, yet value decoding still occurs as a noisy infill problem with right context and generated-history drift. Careful decode recovers many of these rows, so the failure is the clean wave schedule/order, not just base model incapability.
