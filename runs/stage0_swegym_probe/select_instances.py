#!/usr/bin/env python
"""Stage 0 phase 2 PROBE: select + validate ~20 SWE-Gym instances.

For each candidate (stratified across the 10 fork-covered repos, MONAI excluded):
  (a) version present in the fork's MAP_REPO_VERSION_TO_SPECS[repo]  -> scorable
  (b) make_test_spec() succeeds                                     -> eval script buildable
  (c) prebuilt image xingyaoww/sweb.eval.x86_64.<slug_s>:latest EXISTS on Docker Hub
Keep up to N_PER_REPO per repo, TARGET total. Emit subset json + a slug map.
"""
import json, os, sys, urllib.request, urllib.error
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MANIFEST = "data/swe_sft_pool/pool_manifest.json"
TARGET = 20
N_PER_REPO = 2
EXCLUDE_REPOS = {"Project-MONAI/MONAI"}  # not in fork spec map

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as SPECS
from swebench.harness.test_spec import make_test_spec


def slug_s(instance_id: str) -> str:
    # Docker Hub forbids '__'; SWE-Gym publishes with '__' -> '_s_'
    return instance_id.replace("__", "_s_")


def hub_image_exists(slug: str, tag: str = "latest") -> bool:
    repo = f"xingyaoww/sweb.eval.x86_64.{slug}"
    tok_url = (
        "https://auth.docker.io/token?service=registry.docker.io"
        f"&scope=repository:{repo}:pull"
    )
    try:
        with urllib.request.urlopen(tok_url, timeout=30) as r:
            token = json.load(r).get("token", "")
    except Exception as e:  # noqa: BLE001
        print(f"    token err {slug}: {e}", file=sys.stderr)
        return False
    man_url = f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(man_url, method="HEAD")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header(
        "Accept",
        "application/vnd.docker.distribution.manifest.v2+json,"
        "application/vnd.oci.image.manifest.v1+json,"
        "application/vnd.docker.distribution.manifest.list.v2+json,"
        "application/vnd.oci.image.index.v1+json",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception as e:  # noqa: BLE001
        print(f"    manifest err {slug}: {e}", file=sys.stderr)
        return False


def main():
    man = json.load(open(MANIFEST))
    rows = man["instances"]

    # SWE-Gym dataset records for version + test fields
    from datasets import load_dataset
    ds = load_dataset("SWE-Gym/SWE-Gym", split="train")
    rec = {ex["instance_id"]: dict(ex) for ex in ds}

    by_repo = defaultdict(list)
    for r in rows:
        if r["repo"] in EXCLUDE_REPOS:
            continue
        by_repo[r["repo"]].append(r["instance_id"])

    # deterministic order: sort repos by size desc, ids ascending (older/lower num first)
    repos = sorted(by_repo, key=lambda k: (-len(by_repo[k]), k))
    for k in by_repo:
        by_repo[k].sort()

    selected = []
    slugmap = {}
    per_repo_kept = defaultdict(int)
    audit = []

    for repo in repos:
        for iid in by_repo[repo]:
            if len(selected) >= TARGET:
                break
            if per_repo_kept[repo] >= N_PER_REPO:
                break
            ex = rec.get(iid)
            if ex is None:
                continue
            ver = str(ex.get("version"))
            reason = None
            if repo not in SPECS or ver not in SPECS[repo]:
                reason = f"version {ver} not in spec map"
            if reason is None:
                try:
                    make_test_spec(ex)
                except Exception as e:  # noqa: BLE001
                    reason = f"make_test_spec: {type(e).__name__}: {str(e)[:80]}"
            sl = slug_s(iid)
            if reason is None:
                if not hub_image_exists(sl):
                    reason = "no prebuilt xingyaoww image"
            ok = reason is None
            audit.append({"instance_id": iid, "repo": repo, "version": ver,
                          "slug_s": sl, "ok": ok, "reason": reason})
            print(f"  {'KEEP' if ok else 'skip'} {iid} v{ver} {reason or ''}", flush=True)
            if ok:
                selected.append(iid)
                slugmap[iid] = {"slug_s": sl,
                                "xingyaoww": f"xingyaoww/sweb.eval.x86_64.{sl}:latest",
                                "repo": repo, "version": ver}
                per_repo_kept[repo] += 1
        if len(selected) >= TARGET:
            break

    out = {
        "dataset_name": "SWE-Gym/SWE-Gym",
        "split": "train",
        "source": "Stage 0 phase 2 PROBE: 20 SWE-Gym instances, stratified <=2/repo, "
                  "validated (fork spec-covered + make_test_spec ok + prebuilt xingyaoww image)",
        "n": len(selected),
        "instance_ids": selected,
        "slugmap": slugmap,
    }
    os.makedirs("runs/stage0_swegym_probe/artifacts", exist_ok=True)
    json.dump(out, open("runs/stage0_swegym_probe/artifacts/subset_probe20.json", "w"), indent=2)
    json.dump(audit, open("runs/stage0_swegym_probe/artifacts/selection_audit.json", "w"), indent=2)
    print(f"\nSELECTED {len(selected)} instances across "
          f"{len(set(slugmap[i]['repo'] for i in selected))} repos")
    from collections import Counter
    print("per-repo:", dict(Counter(slugmap[i]['repo'] for i in selected)))


if __name__ == "__main__":
    main()
