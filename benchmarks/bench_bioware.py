# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bioware closed-loop benchmark harness

"""Runs the 100-frame bioware × ArcaneZenith closed-loop demo and
records wall time, per-frame latency progression, final identity
drift. Backs `docs/api/bioware.md` §7.1 / §7.2.
"""

import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
    env["MPLBACKEND"] = "Agg"
    demo_path = os.path.join(REPO_ROOT, "examples", "14_bioware_closed_loop_demo.py")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, demo_path],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=600,
    )
    wall = time.perf_counter() - t0
    stdout = proc.stdout.decode()

    # Parse per-frame latencies from the demo's own progress lines:
    #   "Frame 020 | Spikes: ... | Latency: 6916.2 μs"
    frames: list[tuple[int, float]] = []
    for line in stdout.splitlines():
        m = re.search(r"Frame\s+(\d+)\s+\|.*Latency:\s+([\d.]+)", line)
        if m:
            frames.append((int(m.group(1)), float(m.group(2))))

    drift_m = re.search(r"identity drift:\s*([\d.]+)", stdout)
    complete_m = re.search(r"Experiment complete in\s+([\d.]+)\s*s", stdout)
    bursts_m = re.search(r"bursts detected:\s*(\d+)", stdout)

    results = {
        "bioware_closed_loop_100frames": {
            "wall_seconds_driver_side": wall,
            "demo_reported_seconds": float(complete_m.group(1)) if complete_m else None,
            "final_identity_drift": float(drift_m.group(1)) if drift_m else None,
            "total_network_bursts": int(bursts_m.group(1)) if bursts_m else None,
            "per_frame_latency_us": {f"frame_{n:03d}": lat for n, lat in frames},
        }
    }

    print(f"\n{'Metric':<42} {'Value':>18}")
    print("-" * 62)
    if complete_m:
        print(f"{'demo wall (as reported)':<42} {complete_m.group(1):>16} s")
    print(f"{'subprocess wall (driver-observed)':<42} {wall:>16.3f} s")
    if drift_m:
        print(f"{'final identity_drift':<42} {drift_m.group(1):>18}")
    if bursts_m:
        print(f"{'total network bursts detected':<42} {bursts_m.group(1):>18}")
    for frame_num, lat in frames:
        print(f"{'frame ' + str(frame_num) + ' latency (µs)':<42} {lat:>18.1f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_bioware.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
