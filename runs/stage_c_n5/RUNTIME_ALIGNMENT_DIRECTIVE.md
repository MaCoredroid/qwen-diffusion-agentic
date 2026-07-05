# USER DIRECTIVE (2026-07-05): flywheel agent runtime is NOT aligned with official SWE-Verified — FIX ON OUR END
Finding (relayed from the other team): the LumoFlyWheel agent runtime misses the per-instance RUNTIME
DEPENDENCIES official SWE-Verified provides — officially, the agent operates inside the instance image
(repo conda env, deps installed) so it can import the package and RUN TESTS during the episode. Our ported
driver runs the agent in a bare checkout → in-episode test runs / imports fail on missing deps.

CONSEQUENCES FOR OUR DATA:
- The N=5 behavioral numbers (real-edit rates, LOOPING, turn counts) carry this arm-INVARIANT confound:
  agents that try to run tests hit dependency failures and may loop/give up. All arms equally affected →
  the three-arm weights-vs-paradigm ATTRIBUTION remains valid (differential), but absolute behavior +
  resolve rates are NOT official-comparable until fixed.
FIX (ours, do NOT wait for the flywheel team): run each episode INSIDE the official per-instance swebench
docker image (docker now installed; agent working dir = the image's prepared repo env, qwen-code drives it
via the proxy from the host or inside the container), OR materialize the instance conda/venv deps locally
per instance. Prefer the official-image path — it aligns runtime AND scoring in one move.
ACCEPTANCE: agent can `python -c "import <pkg>"` and run the instance's test command in-episode without
dependency errors, on all 5 Tier0 instances, before any re-run is scored.
