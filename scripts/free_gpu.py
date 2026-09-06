"""Kill whatever is holding GPU memory, by asking the GPU rather than by name.

vLLM's EngineCore is a spawned subprocess whose command line does not
contain the launching script's name, so `pkill -f <script>` leaves it
running and holding the entire allocation. The next job then dies with
"Free memory on device cuda:0 (8.24/94.97 GiB) is less than desired". Ask
nvidia-smi which PIDs hold memory and kill those.
"""
import subprocess, sys, time

def holders():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, mem = [x.strip() for x in line.split(",")]
        rows.append((int(pid), mem))
    return rows

keep = {int(p) for p in sys.argv[1:]}
before = holders()
print("holding GPU:", before or "(none)")
for pid, _ in before:
    if pid in keep:
        print(f"  keeping {pid}")
        continue
    subprocess.run(["kill", "-9", str(pid)])
    print(f"  killed {pid}")
time.sleep(8)
print("after:", holders() or "(none)")
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip(), "used")
