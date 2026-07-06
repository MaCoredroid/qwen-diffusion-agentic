# Episode-in-container runtime alignment — design + acceptance

Implements `runs/stage_c_n5/RUNTIME_ALIGNMENT_DIRECTIVE.md`: run each SWE-bench
episode INSIDE the official per-instance swebench image so the agent has the same
prepared runtime official SWE-Verified gives it (repo at `base_commit`,
editable-installed into a conda env with all deps + build artifacts), instead of
our old bare git-worktree checkout where in-episode `import`/tests died on missing
deps. All prior N=5 numbers are deprecated; this is the prerequisite gate.

## Pattern chosen: WORKSPACE-MOUNT + DOCKER-EXEC hybrid

Design axis in the directive was "docker-exec tool wrapper vs workspace mount".
qwen-code has BOTH a shell tool and native file tools (edit/write_file/replace),
so neither pure option alone suffices — we use both, each for what it is good at:

- **Workspace mount (for file tools).** We SEED a host workspace from the image's
  own `/testbed` via `docker cp` — this preserves the editable install linkage and
  build artifacts verbatim (`*.egg-info`, pytest `src/_pytest/_version.py`,
  compiled extensions) — then bind-mount that workspace back over `/testbed` in a
  long-lived per-instance container. qwen-code stays on the HOST with
  `CWD=workspace`, so its native file edits write the mount == the container's
  `/testbed`.
- **Docker-exec (for the shell tool).** Every shell action is routed INTO the
  container via `docker exec` with the official conda-activation preamble
  (`source /opt/miniconda3/bin/activate testbed && cd /testbed && <locale>`), so
  imports/builds/tests use the prepared per-instance env (py3.6/3.9 `testbed`,
  not the base env). For a live qwen-code episode a `bash` PATH-shim
  (`_write_shell_shim`) forwards `run_shell_command` into that same `docker exec`.
- **Patch** = tracked `git diff base_commit` computed inside the container over the
  shared `/testbed` tree.

**Rejected alternative:** bind-mounting a BARE git checkout over `/testbed`. That
shadows the editable install + build artifacts → exactly the "troubled env" the
directive rejects (e.g. pytest fails to import without the generated `_version.py`).
Seeding the mount from the image's own `/testbed` is what avoids this.

Teardown chowns the mount back to the host uid first (the container runs as root
and leaves root-owned `__pycache__`/egg-info), then removes container + workspace.

## Wiring (in `scripts/run_swe_bench_qwen_code.py`)

`--runtime container` (legacy `host` unchanged). New primitives:
`_container_image_for` (`__`→`_1776_` Docker Hub transform), `_seed_workspace_from_image`,
`_start_container`, `_cexec` (conda-activated in-container exec), `_teardown_container`,
`_container_extract_patch`, `_write_shell_shim`, `_run_mock_agent_container`, and
`_process_one_container`. Docker CLI prefix is `SWE_DOCKER_CMD` (default `docker`;
this box uses `sudo -A docker` with `SUDO_ASKPASS` since the docker group is not
active in spawned shells).

## Official image sizes (all present locally; ~19.2 GB total; disk 3.1 T free)

| image | size |
|---|---|
| swebench/sweb.eval.x86_64.django_1776_django-11119 | 3.94 GB |
| swebench/sweb.eval.x86_64.django_1776_django-12754 | 3.95 GB |
| swebench/sweb.eval.x86_64.django_1776_django-13741 | 3.96 GB |
| swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-8399 | 3.60 GB |
| swebench/sweb.eval.x86_64.sympy_1776_sympy-13757 | 3.71 GB |

## Acceptance gate result: 5/5 (see ACCEPTANCE_TABLE.md)

Scripted (NO model) in-episode `import <pkg>` + the exact instance test command,
run through the driver's container primitives. All 5 import with rc 0 and execute
their test command with no dependency errors:

- django x3: `import django` rc 0; `runtests.py … <module>` → "Ran N tests … OK".
- pytest-8399: `import pytest` rc 0; `pytest … test_nose test_unittest` → 59 passed,
  29 skipped. (The 3 "No module named nose/twisted/asynctest" lines are optional
  *test* deps the suite conditionally SKIPs — not runtime dependency errors, so
  non-fatal.)
- sympy-13757: `import sympy` rc 0; `bin/test … test_match test_polytools` →
  168 passed, 5 xfail, 5 exceptions (test-level at base_commit without the fix;
  deps all present).

Evidence in this dir: `acceptance_results.json`, `<instance>__import.log`,
`<instance>__test.log`, `gate_run.log`; full-driver `_process_one_container` path
(mock gold-patch replay through the container → in-container patch extract → mock
eval `resolved`) in `driver_selftest/`.
