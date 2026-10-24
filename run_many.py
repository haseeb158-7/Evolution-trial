import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(seed):
    out = f"data/control/seed_{seed}"
    r = subprocess.run(
        [sys.executable, "-m", "sim.run", "--seed", str(seed), "--out", out],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"seed {seed} done")
    else:
        print(f"seed {seed} FAILED\n{r.stderr[-500:]}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Running {n} seeds on {workers} workers")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(1, n + 1)))