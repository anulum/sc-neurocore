# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia ODE suite benchmark harness

"""Runs the Julia `fusion_solver.jl` `run_benchmarks()` entrypoint and
captures per-solver timing. Backs `docs/api/julia_solvers.md` §7.
"""

import json
import os
import re
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SOLVER_DIR = os.path.join(REPO_ROOT, "src", "sc_neurocore", "accel", "julia", "solvers")


def _resolve_julia() -> str:
    for candidate in (
        os.path.expanduser("~/.juliaup/bin/julia"),
        "/usr/bin/julia",
        "/usr/local/bin/julia",
    ):
        if os.path.exists(candidate):
            return candidate
    return "julia"


def main() -> int:
    julia = _resolve_julia()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [julia, "-e", 'include("fusion_solver.jl"); run_benchmarks()'],
        cwd=SOLVER_DIR,
        capture_output=True,
        timeout=900,
    )
    wall = time.perf_counter() - t0
    stdout = proc.stdout.decode()

    # Lines look like: "§1  LIF ODE                   0.311 ms/solve"
    per_solver: dict[str, float] = {}
    for line in stdout.splitlines():
        m = re.match(r"§(\d+)\s+(.+?)\s+([\d.]+)\s+ms/solve", line)
        if m:
            key = f"§{m.group(1)}_{m.group(2).strip().replace(' ', '_')}"
            per_solver[key] = float(m.group(3))

    results = {
        "julia_solver_suite_wall_seconds": wall,
        "julia_per_solver_ms": per_solver,
    }

    print(f"\n{'Solver':<40} {'ms/solve':>12}")
    print("-" * 54)
    for key, ms in per_solver.items():
        print(f"{key:<40} {ms:>10.3f}")
    print(f"\nFull-suite wall: {wall:.2f} s (includes Julia startup + DiffEq precompile)")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_julia_solvers.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
