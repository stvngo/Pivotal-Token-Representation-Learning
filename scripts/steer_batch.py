#!/usr/bin/env python
"""Run several steer_eval configurations back to back on one GPU.

Equalising every configuration at a common sample size means about a dozen
runs across three models, and launching them one at a time means a human in
the loop for each hand-off. This takes a queue instead, so a session is
given its whole share of the work at once.

Each entry is passed straight through to steer_eval as arguments, and the
queue continues past a failure rather than abandoning the remaining work --
a run that dies takes its own slot down, not the session's.

    scripts/steer_batch.py --spec '[{"--model": "...", ...}, ...]'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

def _find_steer_eval() -> Path:
    """Locate steer_eval.py from wherever this script was dropped.

    colab_job uploads the batch script to the repo root while the repo's own
    copy of steer_eval.py sits in scripts/, so resolving next to __file__
    alone finds nothing and python exits 2 before printing anything useful.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "steer_eval.py", here / "scripts" / "steer_eval.py",
                 here.parent / "scripts" / "steer_eval.py"):
        if cand.exists():
            return cand
    raise SystemExit(f"steer_eval.py not found near {here}")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="JSON list of arg dicts")
    a = ap.parse_args()

    runs = json.loads(a.spec)
    print(f"[batch] {len(runs)} runs queued", flush=True)
    results = []
    for i, cfg in enumerate(runs, 1):
        argv = [sys.executable, str(_find_steer_eval())]
        for k, v in cfg.items():
            argv.append(k)
            if v is not None:
                argv.append(str(v))
        label = cfg.get("--out", f"run{i}")
        print(f"\n[batch {i}/{len(runs)}] {Path(label).name}", flush=True)
        print(f"[batch] {' '.join(argv[2:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(argv).returncode
        results.append((label, rc, round(time.time() - t0, 1)))
        print(f"[batch {i}/{len(runs)}] rc={rc} in {results[-1][2]}s", flush=True)

    print("\n[batch] summary")
    for label, rc, secs in results:
        print(f"  {'ok ' if rc == 0 else 'FAIL'} {Path(label).name:<42} {secs:>8.1f}s", flush=True)
    print(f"[batch] {sum(1 for _, rc, _ in results if rc == 0)}/{len(results)} succeeded",
          flush=True)


if __name__ == "__main__":
    main()
