# USER DECISION (2026-07-05): SWE-Verified eval runs LOCALLY — do NOT use the alienware x86 offload path
Binding for Stage C (N=5 and N=25-50): patch application + test execution + resolve@1 scoring happen on
THIS machine (local docker / local swebench harness / the flywheel's local eval classifier — whichever
works locally). The swe_x86_helpers offload scripts (offload_codex_proxy.sh / relaunch_proxy_remote.sh)
are OUT OF SCOPE by user decision. If local eval infra is missing pieces, set them up locally and document
the setup in the driver README (this becomes part of the reproducible Stage-C recipe).
