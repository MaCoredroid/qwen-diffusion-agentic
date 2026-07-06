#!/usr/bin/env bash
# W2 FRESH diffusion boot-probe (frozen serving: gmu 0.74, c=4, mml 32768) at
# max long-context load, run BEFORE the diffusion arm per directive. Reuses the
# gate's boot_probe_diffusion.sh + boot_probe_client.py verbatim; result copied
# into runs/w2_n50/boot_probe_result.json so the gate's frozen probe stays intact.
set -uo pipefail
cd /home/mark/qwen_diffusion
export SUDO_ASKPASS="${SUDO_ASKPASS:?export SUDO_ASKPASS}"
GMU=0.74; C=4; MML=32768; PTOK="${1:-12000}"
bash runs/stage_c_n5v3_gate/boot_probe_diffusion.sh "$GMU" "$C" "$MML" "$PTOK"
RC=$?
SRC="runs/stage_c_n5v3_gate/boot_probe_result_g${GMU}_c${C}.json"
DST="runs/w2_n50/boot_probe_result.json"
if [[ -s "$SRC" ]]; then cp "$SRC" "$DST"; echo "[w2-probe] copied -> $DST"; .venv/bin/python -c "import json;d=json.load(open('$DST'));print('boot_ok',d.get('boot_ok'),'no_alloc_fail',d.get('no_allocation_failure'),'peak',d.get('peak_used_mib'),'headroom',d.get('headroom_mib'))"; fi
exit $RC
