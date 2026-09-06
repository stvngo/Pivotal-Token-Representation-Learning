#!/usr/bin/env python
"""Run a long job on a Colab VM detached, and follow its log.

`colab exec` streams output over a websocket and gives up after a few
minutes, which is fine for a probe but not for anything real: vLLM alone
spends minutes compiling for a new GPU architecture before it generates a
token, and a PTS run is hours. So the job is started with `nohup` on the VM
and its log is polled instead.

    # start, and follow until it exits
    scripts/colab_job.py -s ptrl-g4 --script scripts/pts_run.py -- --model Qwen/Qwen3-0.6B

    # start and detach; check on it later
    scripts/colab_job.py -s ptrl-g4 --script job.py --no-follow
    scripts/colab_job.py -s ptrl-g4 --tail
    scripts/colab_job.py -s ptrl-g4 --status
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = str(ROOT / "scripts" / "colab")
REMOTE_ROOT = "/content/ptrl"
LOG = "/content/job.log"
PIDFILE = "/content/job.pid"


def colab_exec(session: str, code: str, *, quiet: bool = True) -> str:
    """Run Python on the VM and return its stdout."""
    proc = subprocess.run(
        [COLAB, "exec", "-s", session],
        input=code,
        capture_output=True,
        text=True,
    )
    out = proc.stdout or ""
    if not quiet and proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
    # Strip the CLI's own chatter and terminal control bytes.
    lines = [ln for ln in out.splitlines() if not ln.startswith("[colab]")]
    return "\n".join(lines)


def push_token(session: str) -> bool:
    """Write HF_TOKEN into the remote repo's .env.

    Not passed on the command line or through the Popen environment, since
    both are visible in `ps` on the VM. A 0600 file in the repo root is
    where resolve_hf_token already looks.
    """
    sys.path.insert(0, str(ROOT))
    from probe_pipeline.artifacts_io import resolve_hf_token

    token = resolve_hf_token()
    if not token:
        return False
    code = f"""
import os, pathlib
p = pathlib.Path({REMOTE_ROOT!r}) / ".env"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("HF_TOKEN=" + {token!r} + chr(10))
os.chmod(p, 0o600)
print("token written")
"""
    return "token written" in colab_exec(session, code)


def start(session: str, script: str, args: list[str], *, sync: bool) -> None:
    local = Path(script)
    if not local.exists():
        sys.exit(f"no such script: {script}")

    if sync:
        print(f"[job] syncing repo on {session}")
        print(colab_exec(session, SYNC_CODE))
    # After syncing, since a fresh clone would not carry it.
    print(f"[job] hf token: {'ok' if push_token(session) else 'not available'}")

    remote = f"{REMOTE_ROOT}/{local.name}" if str(local).startswith("scripts/") else f"/content/{local.name}"
    payload = local.read_text()
    argstr = " ".join(shlex.quote(a) for a in args)

    # Build the shell command first and embed it as a *repr*. Interpolating
    # argstr straight into a double-quoted string literal in the generated
    # code breaks the moment an argument contains a double quote -- a JSON
    # spec, say -- and the failure is silent: the generated source raises a
    # SyntaxError on the VM and colab_exec simply returns nothing.
    cmd = (f"cd {REMOTE_ROOT} && nohup python -u {remote} {argstr} "
           f"> {LOG} 2>&1 & echo $! > {PIDFILE}")
    code = f"""
import os, subprocess
os.makedirs({REMOTE_ROOT!r}, exist_ok=True)
open({remote!r}, "w").write({payload!r})
# start_new_session so the process survives this kernel call returning
p = subprocess.Popen({cmd!r}, shell=True, start_new_session=True)
p.wait()
print("started pid", open({PIDFILE!r}).read().strip())
"""
    print(colab_exec(session, code))


SYNC_CODE = f"""
import subprocess, os
os.chdir("/content")
if os.path.isdir("{REMOTE_ROOT}/.git"):
    c = "cd {REMOTE_ROOT} && git fetch -q origin && git reset -q --hard origin/rebuild/pipeline-and-scale"
else:
    c = ("rm -rf {REMOTE_ROOT} && git clone -q --branch rebuild/pipeline-and-scale "
         "https://github.com/stvngo/Pivotal-Token-Representation-Learning.git {REMOTE_ROOT}")
subprocess.run(c, shell=True)
print(subprocess.run("cd {REMOTE_ROOT} && git log --oneline -1", shell=True,
                     capture_output=True, text=True).stdout.strip())
"""


def status(session: str) -> tuple[bool, str]:
    code = f"""
import os, subprocess
alive = False
try:
    pid = int(open({PIDFILE!r}).read().strip())
    os.kill(pid, 0)
    alive = True
except Exception:
    pass
print("ALIVE" if alive else "DONE")
"""
    out = colab_exec(session, code)
    return "ALIVE" in out, out


def tail(session: str, n: int = 40) -> str:
    code = f"""
import subprocess
print(subprocess.run(["tail","-n","{n}","{LOG}"], capture_output=True, text=True).stdout)
"""
    return colab_exec(session, code)


def follow(session: str, interval: int = 30) -> None:
    seen = 0
    while True:
        alive, _ = status(session)
        text = tail(session, 200)
        if len(text) > seen:
            sys.stdout.write(text[seen:])
            sys.stdout.flush()
            seen = len(text)
        if not alive:
            print("\n[job] finished")
            return
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", required=True)
    ap.add_argument("--script", help="local script to run on the VM")
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--no-sync", action="store_true", help="skip the git pull on the VM")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("args", nargs="*", help="arguments forwarded to the script")
    a = ap.parse_args()

    if a.status:
        print(status(a.session)[1])
        return
    if a.tail:
        print(tail(a.session, a.n))
        return
    if not a.script:
        ap.error("--script is required unless --tail/--status")

    start(a.session, a.script, a.args, sync=not a.no_sync)
    if not a.no_follow:
        follow(a.session)


if __name__ == "__main__":
    main()
