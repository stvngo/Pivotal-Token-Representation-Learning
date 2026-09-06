#!/usr/bin/env python
"""Re-adopt every orphaned VM under its previous name, unattended.

Pruning happens every few minutes during long runs, and each prune also
orphans the keep-alive daemon -- without which Colab reclaims the VM a few
minutes later. Handling that by hand is mechanical and easy to be slow
about, and being slow about it loses the machine.

Names matter because monitoring and job control address sessions by name,
and the CLI otherwise re-adopts under whatever stale name it has lying
around. So the endpoint-to-name mapping is recorded in a small state file
and reapplied here.

    scripts/reattach_all.py --learn        # record current name<-endpoint
    scripts/reattach_all.py                # re-adopt any orphans
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "artifacts" / "session_map.json"


def load_map() -> dict[str, str]:
    return json.loads(MAP.read_text()) if MAP.exists() else {}


def save_map(m: dict[str, str]) -> None:
    MAP.parent.mkdir(parents=True, exist_ok=True)
    MAP.write_text(json.dumps(m, indent=2))


def listing() -> list[tuple[str, str]]:
    """(state, endpoint) for every assignment the server knows about."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "colab_reattach.py"),
                          "--list"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        if line.startswith("[") and "|" in line:
            state = line[1:line.index("]")]
            endpoint = line[line.index("]") + 1:].split("|")[0].strip()
            rows.append((state, endpoint))
    return rows


def local_names() -> dict[str, str]:
    """endpoint -> local session name, from the CLI's own listing."""
    out = subprocess.run([str(ROOT / "scripts" / "colab"), "sessions"],
                         capture_output=True, text=True).stdout
    m = {}
    for line in out.splitlines():
        if line.startswith("[") and "|" in line:
            name = line[1:line.index("]")]
            endpoint = line[line.index("]") + 1:].split("|")[0].strip()
            if name != "?":
                m[endpoint] = name
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learn", action="store_true",
                    help="record the current endpoint->name mapping and exit")
    a = ap.parse_args()

    if a.learn:
        m = load_map() | local_names()
        save_map(m)
        for e, n in m.items():
            print(f"  {n:<12} {e}")
        print(f"wrote {MAP}")
        return

    m = load_map()
    orphans = [e for state, e in listing() if state == "ORPHAN"]
    if not orphans:
        print("no orphans")
        return
    for e in orphans:
        name = m.get(e)
        if not name:
            print(f"  {e}: no recorded name; skipping (adopt it by hand once, "
                  f"then --learn)")
            continue
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "colab_reattach.py"),
                            "--name", name, "--endpoint", e],
                           capture_output=True, text=True)
        ok = "adopted" in r.stdout
        keep = "keep-alive respawned" in r.stdout
        print(f"  {name:<12} {'adopted' if ok else 'FAILED'}"
              f"{', keep-alive respawned' if keep else ', KEEP-ALIVE NOT RESTARTED'}")
        if not ok:
            print("   ", (r.stdout + r.stderr).strip()[:200])


if __name__ == "__main__":
    main()
