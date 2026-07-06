#!/usr/bin/env python3
"""ACCEPTANCE GATE for the RUNTIME_ALIGNMENT_DIRECTIVE (2026-07-05).

Hard gate BEFORE any scored episode: on all 5 Tier0 instances, running INSIDE the
official per-instance swebench image, prove that
  (1) `python -c "import <pkg>"`  succeeds (rc 0), and
  (2) the instance's exact test command runs to completion (tests execute)
      WITHOUT dependency errors,
using the SAME container mechanism the driver uses for episodes (seed workspace
from the image /testbed, bind-mount it back, route shell via `docker exec` with
the official conda `testbed` activation). NO model is involved -- these are
scripted tool calls that stand in for the agent's shell tool.

Run with the swebench-bearing venv (for test-command construction) e.g.:
  SWE_DOCKER_CMD='sudo -A docker' SUDO_ASKPASS=.../askpass.sh \
    runs/stage_c_n5/local_eval/.venv-harness/bin/python \
    runs/stage_c_n5/acceptance/acceptance_gate.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path("/home/mark/qwen_diffusion")
LE = REPO / "runs/stage_c_n5/local_eval"
OUT = REPO / "runs/stage_c_n5/acceptance"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
import run_swe_bench_qwen_code as D  # noqa: E402

# swebench (harness venv) builds the exact official test command per instance.
from swebench.harness.test_spec.python import get_test_directives  # noqa: E402
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # noqa: E402

INST = json.load(open(LE / "instances.json"))
PKG = {"django/django": "django", "pytest-dev/pytest": "pytest", "sympy/sympy": "sympy"}

# "tests actually ran" markers (framework loaded + executed) -> deps are present.
RAN_MARKERS = [
    r"Ran \d+ test",                       # django / unittest
    r"\d+ passed", r"\d+ failed", r"\d+ error", r"\d+ skipped",  # pytest / sympy summary
    r"tests finished", r"test process starts", r"passed in ", r"= ERRORS =",
]
# top-level missing-dependency signatures (fatal only if tests never ran).
DEP_ERR_MARKERS = [
    "ModuleNotFoundError", "No module named", "ImportError while loading conftest",
    "cannot import name", "DistributionNotFound", "Could not find a version",
]


def prep(rec):
    r = dict(rec)
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(r[k], str):
            r[k] = json.loads(r[k])
    r["version"] = str(r["version"])
    return r


def test_command(r):
    spec = MAP_REPO_VERSION_TO_SPECS[r["repo"]][r["version"]]
    tc = spec["test_cmd"]
    tc = tc[-1] if isinstance(tc, list) else tc
    return tc + " " + " ".join(get_test_directives(r))


def any_re(patterns, text):
    return any(re.search(p, text) for p in patterns)


def any_sub(subs, text):
    return any(s in text for s in subs)


def run_instance(iid):
    r = prep(INST[iid])
    pkg = PKG[r["repo"]]
    base = r["base_commit"]
    image = D._container_image_for(iid)
    container = f"accept_{iid.replace('__', '_')}"
    ws = OUT / "ws" / iid
    row = {"instance_id": iid, "repo": r["repo"], "version": r["version"], "pkg": pkg,
           "image": image, "container": container}
    t0 = time.time()
    try:
        if not D._image_present(image):
            row.update(accepted=False, error=f"image_absent:{image}")
            return row
        D._seed_workspace_from_image(image=image, workspace=ws)
        D._start_container(name=container, image=image, workspace=ws)

        # (1) import <pkg> IN-EPISODE (conda testbed activated)
        imp = D._cexec(container, f"python -c 'import {pkg}; print({pkg})'", timeout=300)
        import_rc = imp.returncode
        import_out = (imp.stdout or "") + (imp.stderr or "")
        (OUT / f"{iid}__import.log").write_text(
            f"$ python -c 'import {pkg}'\nrc={import_rc}\n{import_out}\n", encoding="utf-8")

        # (2) instance test command IN-EPISODE (base_commit, no fix/test_patch)
        cmd = test_command(r)
        row["test_cmd"] = cmd
        tst = D._cexec(container, cmd, timeout=1800)
        test_rc = tst.returncode
        test_out = (tst.stdout or "") + "\n===STDERR===\n" + (tst.stderr or "")
        (OUT / f"{iid}__test.log").write_text(
            f"$ {cmd}\nrc={test_rc}\n{test_out}\n", encoding="utf-8")

        test_ran = any_re(RAN_MARKERS, test_out)
        dep_err = any_sub(DEP_ERR_MARKERS, test_out)
        fatal_dep = dep_err and not test_ran

        import_ok = import_rc == 0
        accepted = bool(import_ok and test_ran and not fatal_dep)
        row.update(
            import_rc=import_rc, import_ok=import_ok, import_tail=import_out.strip()[-300:],
            test_rc=test_rc, test_ran=test_ran, dep_error_signature=dep_err,
            fatal_dependency_error=fatal_dep, accepted=accepted,
            test_tail=test_out.strip()[-500:], elapsed_s=round(time.time() - t0, 1),
        )
        return row
    except Exception as exc:  # noqa: BLE001
        import traceback
        row.update(accepted=False, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-1200:], elapsed_s=round(time.time() - t0, 1))
        return row
    finally:
        # robust teardown: chown root-created test artifacts back before removal
        D._teardown_container(container, ws)


def main():
    ids = list(INST.keys())
    if len(sys.argv) > 1:
        ids = [x for x in sys.argv[1].split(",") if x.strip()]
    rows = []
    for iid in ids:
        print(f">>> {iid} ...", flush=True)
        row = run_instance(iid)
        rows.append(row)
        json.dump(rows, open(OUT / "acceptance_results.json", "w"), indent=2)
        print(f"    accepted={row.get('accepted')} import_rc={row.get('import_rc')} "
              f"test_ran={row.get('test_ran')} fatal_dep={row.get('fatal_dependency_error')} "
              f"({row.get('elapsed_s')}s)", flush=True)

    n_ok = sum(1 for r in rows if r.get("accepted"))
    # markdown table
    lines = [
        "# Runtime-alignment ACCEPTANCE GATE",
        "",
        "Scripted (NO model) in-episode checks INSIDE the official per-instance "
        "swebench image: `import <pkg>` + the exact instance test command, via the "
        "driver's container mechanism (workspace seeded from the image /testbed, "
        "bind-mounted, shell routed through `docker exec` with conda `testbed` "
        "activation).",
        "",
        f"**Result: {n_ok}/{len(rows)} accepted**",
        "",
        "| instance | pkg | import rc | tests ran | dep-error | accepted |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance_id']} | {r.get('pkg','?')} | {r.get('import_rc','?')} "
            f"| {'yes' if r.get('test_ran') else 'NO'} "
            f"| {'FATAL' if r.get('fatal_dependency_error') else 'none'} "
            f"| {'PASS' if r.get('accepted') else 'FAIL'} |")
    lines += ["", "Evidence: `acceptance_results.json`, `<instance>__import.log`, "
              "`<instance>__test.log` in this directory.", ""]
    (OUT / "ACCEPTANCE_TABLE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== ACCEPTANCE {n_ok}/{len(rows)} ===")
    print("\n".join(lines[-(len(rows) + 4):]))
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
