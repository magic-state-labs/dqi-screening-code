"""Run script 30 (key-less portfolio) or 36 (key holder) for the s=6 McEliece family, one job, logging peak memory.

usage: python runs/run_ext6.py {30|36} <size> <seed> [--flag value ...]
Extra flags replace the defaults below (e.g. --line-seconds 1800 --tag ext6bud1800); the log name carries
the tag, so runs with different tags do not overwrite each other.
"""
import subprocess
import sys
import time

import psutil

script, size, seed = sys.argv[1], sys.argv[2], sys.argv[3]
if script == "30":
    base = [sys.executable, "scripts/30_alternant_trapdoor_finite_control.py"]
    flags = {"--ext": "6", "--alpha": "0.08", "--sizes": size, "--seeds": seed,
             "--line-seconds": "600", "--heat-seconds": "600", "--tag": "ext6bud600"}
else:
    base = [sys.executable, "scripts/36_keyholder_classical_control.py"]
    flags = {"--ext": "6", "--alpha": "0.08", "--sizes": size, "--seeds": seed, "--long-seconds": "0",
             "--warm-seconds": "600", "--keyless-tag": "ext6bud600", "--tag": "ext6warm600"}
extra = sys.argv[4:]
for name, value in zip(extra[::2], extra[1::2]):
    flags[name] = value  # later flags replace the defaults (the scripts read the first occurrence)
cmd = base + [x for kv in flags.items() for x in kv]
tag = flags["--tag"]
default_tag = "ext6bud600" if script == "30" else "ext6warm600"
suffix = "" if tag == default_tag else f"-{tag}"
log = f"results/logs-bud600/ext6-{script}-m{size}-s{seed}{suffix}.log"
start = time.time()
with open(log, "w", encoding="utf-8") as out:
    out.write(" ".join(cmd[1:]) + "\n")
    out.flush()
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, psutil.Process(proc.pid).memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(5)
    out.write(f"\n[run_ext6] exit {proc.returncode} peak_rss_MB {peak / 2**20:.0f} seconds {time.time() - start:.0f}\n")
print(log)
