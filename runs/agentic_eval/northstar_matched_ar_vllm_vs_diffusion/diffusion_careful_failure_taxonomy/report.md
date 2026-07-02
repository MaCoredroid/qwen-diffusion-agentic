# Diffusion-Careful Matched-20 Failure Taxonomy

Scope: 29 diffusion-careful misses from the matched-20 generated-history eval (`34/63` exact-args).
A gold-prior-history counterfactual reruns each missed turn with the same careful decoder; if it becomes exact, the miss is counted as generated-history compounding.
Never-train careful misses were skipped: no never-train diffusion-careful row exists yet, so this was not a quick taxonomy add-on.

## Split

| class | count |
|---|---:|
| format_or_schema_error | 9 |
| generated_history_compounding | 5 |
| missing_extra_or_wrong_call | 1 |
| stop_or_truncation | 4 |
| wrong_value_content | 10 |

## Symptom Flags

| symptom | count |
|---|---:|
| format_or_schema_error | 15 |
| missing_extra_or_wrong_call | 8 |
| stop_or_truncation | 4 |
| wrong_value_content | 10 |

## Reward Implication

Primary miss mass is {'format_or_schema_error': 9, 'generated_history_compounding': 5, 'missing_extra_or_wrong_call': 1, 'stop_or_truncation': 4, 'wrong_value_content': 10}. Use a ToolRL-style graded reward with explicit format/schema/name/arg-name/value terms; add episode-level credit for generated-history compounding (5 turns), and keep exact-args as the promotion gate. Train-fixable direct current-turn misses: 20/29.

## Rows

| episode | turn | class | gold-history exact | symptom flags | first mismatch | value shapes |
|---|---:|---|---:|---|---|---|
| heldout_seed_run1clean_0000 | 1 | wrong_value_content | 0 | wrong_value_content | argument_value: apply_gates.gates gold="[{'gate': 'H', 'target': [0]}, {'gate': " pred="[{'gate': 'h', 'target': [0]}]" | {"string": 2} |
| heldout_seed_run1clean_0000 | 2 | generated_history_compounding | 1 | missing_extra_or_wrong_call | tool_name: gold=run_circuit pred=apply_gates | {"string": 2} |
| heldout_seed_run1clean_0000 | 3 | generated_history_compounding | 1 | missing_extra_or_wrong_call | tool_name: gold=visualize_quantum_state pred=run_circuit | {"string": 1} |
| heldout_seed_run1clean_0001 | 0 | format_or_schema_error | 0 | format_or_schema_error | argument_value: categorize_parties.parties gold="[{'name': 'Emma Johnson', 'role': 'Plain" pred='None' | {"string": 2} |
| heldout_seed_run1clean_0001 | 1 | format_or_schema_error | 0 | format_or_schema_error | argument_value: index_documents.documents gold="[{'title': 'Complaint', 'type': 'Legal B" pred='None' | {"string": 2} |
| heldout_seed_run1clean_0008 | 0 | stop_or_truncation | 0 | format_or_schema_error,missing_extra_or_wrong_call,stop_or_truncation | call_count: gold=1 pred=0 | {"string": 2} |
| heldout_seed_run1clean_0008 | 1 | stop_or_truncation | 0 | format_or_schema_error,missing_extra_or_wrong_call,stop_or_truncation | call_count: gold=1 pred=0 | {"string": 2} |
| heldout_seed_run1clean_0008 | 2 | stop_or_truncation | 0 | format_or_schema_error,missing_extra_or_wrong_call,stop_or_truncation | call_count: gold=1 pred=0 | {"string": 2} |
| heldout_seed_run1clean_0009 | 0 | format_or_schema_error | 0 | format_or_schema_error | argument_value: fetch_historical_stock_prices.tickers gold="['AAPL', 'MSFT', 'GOOGL']" pred='None' | {"string": 3} |
| heldout_seed_run1clean_0009 | 1 | wrong_value_content | 0 | wrong_value_content | argument_value: calculate_future_returns.scenarios gold="[{'name': 'Optimistic', 'growth_rate': 1" pred="[{'name': 'optimistic', 'growth_rate': 1" | {"string": 2} |
| heldout_seed_run1clean_0009 | 2 | format_or_schema_error | 0 | format_or_schema_error | argument_value: assess_portfolio_risk.portfolio gold="[{'ticker': 'AAPL', 'weight': 0.333}, {'" pred='None' | {"string": 2} |
| heldout_seed_run1clean_0010 | 0 | wrong_value_content | 0 | wrong_value_content | argument_value: plan_project.project_scope gold='Reengineering finance and accounting wor' pred='Reengineering finance and accounting acc' | {"string": 3} |
| heldout_seed_run1clean_0010 | 2 | format_or_schema_error | 0 | format_or_schema_error | argument_value: track_progress.milestones gold="['Workflow Analysis Completion', 'System" pred='None' | {"string": 2} |
| heldout_seed_run1clean_0010 | 3 | format_or_schema_error | 0 | format_or_schema_error | argument_value: generate_report.include_sections gold="['Executive Summary', 'Milestone Achieve" pred='None' | {"string": 3} |
| heldout_seed_run1clean_0015 | 1 | wrong_value_content | 0 | wrong_value_content | argument_value: add_device_to_home_assistant.device_id gold='device_002' pred='device_001' | {"string": 2} |
| heldout_seed_run1clean_0018 | 0 | format_or_schema_error | 0 | format_or_schema_error | argument_value: recommend_content_based_on_viewing_habits.viewing_history gold="[{'content_id': 'movie_201', 'watched_on" pred='None' | {"string": 3} |
| heldout_seed_run1clean_0019 | 0 | format_or_schema_error | 0 | format_or_schema_error | argument_value: control_lighting.lighting_scene gold='Movie Night' pred='None' | {"string": 2} |
| heldout_seed_run1clean_0023 | 0 | stop_or_truncation | 0 | format_or_schema_error,missing_extra_or_wrong_call,stop_or_truncation | call_count: gold=1 pred=0 | {"string": 3} |
| heldout_seed_run1clean_0023 | 1 | missing_extra_or_wrong_call | 0 | missing_extra_or_wrong_call | tool_name: gold=manage_venue pred=book_artist | {"string": 3} |
| heldout_seed_run1clean_0023 | 2 | generated_history_compounding | 1 | missing_extra_or_wrong_call | tool_name: gold=schedule_event pred=manage_venue | {"string": 2} |
| heldout_seed_run1clean_0026 | 0 | format_or_schema_error | 0 | format_or_schema_error | argument_value: update_customer_profile.profile_updates gold="{'address': '1234 Telecom Lane, Datacity" pred='None' | {"string": 2} |
| heldout_seed_run1clean_0026 | 1 | generated_history_compounding | 1 | format_or_schema_error | argument_value: track_service_usage.end_date gold='2023-03-31' pred='None' | {"string": 3} |
| heldout_seed_run1clean_0026 | 2 | generated_history_compounding | 1 | format_or_schema_error | argument_value: manage_support_tickets.resolution_notes gold='Replaced faulty router, service restored' pred='None' | {"string": 3} |
| heldout_seed_run1clean_0030 | 0 | wrong_value_content | 0 | wrong_value_content | argument_value: analyze_customer_data.customer_data_source gold='CRM System' pred='CRM' | {"string": 2} |
| heldout_seed_run1clean_0030 | 1 | wrong_value_content | 0 | wrong_value_content | argument_value: segment_audience.analysis_results gold='Analysis Results Placeholder' pred='heldout_seed_run1clean_0030_turn_0' | {"string": 2} |
| heldout_seed_run1clean_0030 | 2 | wrong_value_content | 0 | wrong_value_content | argument_value: create_targeted_ad_campaign.campaign_parameters gold="{'budget': 5000, 'start_date': '2023-05-" pred="{'budget': 5000, 'start_date': '2024-05-" | {"string": 2} |
| heldout_seed_run1clean_0031 | 0 | wrong_value_content | 0 | wrong_value_content | argument_value: manage_artist_lineup.festival_id gold='Soundwave' pred='SW2023' | {"string": 2} |
| heldout_seed_run1clean_0031 | 1 | wrong_value_content | 0 | wrong_value_content | argument_value: track_ticket_sales.festival_id gold='Soundwave' pred='SW2023' | {"string": 2} |
| heldout_seed_run1clean_0031 | 2 | wrong_value_content | 0 | wrong_value_content | argument_value: coordinate_event_logistics.festival_id gold='Soundwave' pred='SW2023' | {"string": 2} |
