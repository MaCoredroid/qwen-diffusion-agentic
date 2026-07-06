# HANDOFF (monitor, 2026-07-06 04:20): the prior run agent died mid-debug — its findings
1. The 2-episode concurrent AR smoke DOUBLE-RAN the same instance: the worker claim logic has a real
   non-deterministic race (reproduced in isolation: trials 1-3 double-run, 4-5 correct). Both slots got
   success from the mkdir-based claim — check whether the claim uses `mkdir -p` (which succeeds on EEXIST
   and is therefore NOT atomic-exclusive) or races on a pre-check; fix = plain `mkdir` (fails on EEXIST) or
   `os.mkdir` try/except as the sole claim, no pre-checks.
2. Fix the claim, RE-RUN the 2-episode smoke (must claim distinct instances across ≥5 trials), THEN launch
   the full AR arm. Everything else (pool manifest, 50 images, detached pattern, boot timing ~81s) is done
   and valid — do not redo.
