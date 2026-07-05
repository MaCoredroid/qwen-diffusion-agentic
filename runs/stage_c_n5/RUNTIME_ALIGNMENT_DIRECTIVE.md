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

## UPDATE (user, 2026-07-05): N=5 numbers DEPRECATED, not caveated
"Redo the numbers after fixing — a troubled env can't measure real intelligence." ALL N=5 behavioral
numbers (both arms + any partial third-arm data) are DEPRECATED for any claim. The disambiguation workflow
was STOPPED mid-flight to avoid spending GPU on discarded data. Order of work: (1) runtime alignment
(episodes inside official per-instance swebench images, acceptance test on all 5), (2) clean THREE-ARM
re-run (stock-AR / merged-AR / diffusion) on the aligned runtime = the true behavioral baseline,
(3) N=25-50 go/no-go from THAT data.
