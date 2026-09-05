#!/usr/bin/env python
"""Re-adopt a running Colab VM whose local session record was lost.

The CLI treats a 401/404 on `exec` as "the backend pruned the VM" and
deletes its local record. That inference is sometimes wrong -- a transient
auth failure on a long run produces the same symptom while the VM is very
much alive, still running the job. `colab sessions` then shows it as an
orphan (`[?]`) and no command can address it by name, so the work on it is
unreachable.

Everything needed to rebuild the record comes back from the server's own
assignment listing: `runtimeProxyInfo` carries the VM token and URL
alongside the endpoint and accelerator. This reconstructs the
`SessionState` and writes it back.

    scripts/colab_reattach.py --list
    scripts/colab_reattach.py --name ptrl-g4 --endpoint gpu-g4-s-...

With one orphan and no `--endpoint`, it adopts that one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ensure_certs() -> None:
    """Same fix scripts/colab applies: python.org Python ships no CA bundle."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        out = subprocess.run(
            [str(ROOT / "venv" / "bin" / "python"), "-c", "import certifi;print(certifi.where())"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            os.environ["SSL_CERT_FILE"] = out.stdout.strip()
            os.environ.setdefault("REQUESTS_CA_BUNDLE", out.stdout.strip())
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="local name to adopt the VM under")
    ap.add_argument("--endpoint", default=None, help="which assignment, if several")
    ap.add_argument("--list", action="store_true", help="show server-side assignments and exit")
    a = ap.parse_args()

    ensure_certs()
    from colab_cli.common import state          # noqa: E402
    from colab_cli.state import SessionState    # noqa: E402

    assignments = state.client.list_assignments()
    if not assignments:
        sys.exit("no assignments on the server; the VM is really gone")

    known = {s.endpoint for s in state.store.list().values()}
    if a.list:
        for asg in assignments:
            tag = "tracked" if asg.endpoint in known else "ORPHAN"
            print(f"[{tag}] {asg.endpoint} | {asg.accelerator.value} | {asg.variant.name}")
        return

    orphans = [x for x in assignments if x.endpoint not in known]
    if a.endpoint:
        chosen = [x for x in assignments if x.endpoint == a.endpoint]
        if not chosen:
            sys.exit(f"no assignment with endpoint {a.endpoint}")
        target = chosen[0]
    elif len(orphans) == 1:
        target = orphans[0]
    elif not orphans:
        sys.exit("no orphans: every assignment already has a local record")
    else:
        sys.exit(
            "several orphans; pass --endpoint:\n  "
            + "\n  ".join(x.endpoint for x in orphans)
        )

    name = a.name or "recovered"
    if state.store.get(name) is not None:
        sys.exit(f"a session named {name!r} already exists; pick another --name")

    # kernel_id is left unset on purpose: the CLI reconnects to the running
    # kernel on the next exec. Nothing about the VM's filesystem is touched,
    # so a job still running under nohup keeps running.
    state.store.add(
        SessionState(
            name=name,
            token=target.runtime_proxy_info.token,
            url=target.runtime_proxy_info.url,
            endpoint=target.endpoint,
            variant=target.variant.name,
            accelerator=target.accelerator.value,
        )
    )
    print(f"adopted {target.endpoint} ({target.accelerator.value}) as '{name}'")
    print(f"check it with:  scripts/colab status -s {name}")


if __name__ == "__main__":
    main()
