# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — formal proofs benchmark harness

"""Times `lean` elaboration of `safety_bounds.lean` and counts the
proved-vs-axiomatised split. Backs `docs/api/formal.md` §7.
"""

import json
import os
import re
import hashlib
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROOF_FILE = os.path.join(
    REPO_ROOT, "src", "sc_neurocore", "formal", "proofs", "safety_bounds.lean"
)
EXPECTED_AXIOMS = [
    "sc_precision_numerator_bound",
    "sc_add_preserves_range",
]
EXPECTED_THEOREMS = [
    "halt_triggered_complete",
    "monitor_soundness",
    "safe_of_halt_false",
    "halt_false_of_safe",
    "unsafe_of_halt_true",
    "halt_true_of_unsafe",
    "safe_transition",
    "safe_transition_halt_deasserted",
    "lif_spike_resets",
    "lif_integrate_clips",
    "lif_spike_branch_bounded",
    "lif_integrate_branch_bounded",
    "lif_threshold_preserved",
    "lif_v_max_preserved",
    "lif_v_reset_preserved",
    "lif_reset_bound_preserved",
    "lif_membrane_bounded",
    "lif_next_membrane_bounded",
    "scc_bounded",
    "scc_left_bounded",
    "scc_right_bounded",
]


def _resolve_lean() -> str:
    for candidate in (
        os.path.expanduser("~/.elan/bin/lean"),
        "/usr/bin/lean",
        "/usr/local/bin/lean",
    ):
        if os.path.exists(candidate):
            return candidate
    return "lean"


def main() -> int:
    lean = _resolve_lean()

    # Cold-ish timing (Lean toolchain is typically already resident on elan).
    t0 = time.perf_counter()
    proc = subprocess.run([lean, PROOF_FILE], capture_output=True, timeout=300)
    wall = time.perf_counter() - t0

    # Warm run (file already in page cache).
    t1 = time.perf_counter()
    proc2 = subprocess.run([lean, PROOF_FILE], capture_output=True, timeout=300)
    wall_warm = time.perf_counter() - t1

    # Count theorem declarations and explicit top-level axioms from the source file.
    with open(PROOF_FILE, "r") as fh:
        src = fh.read()
    theorem_names = re.findall(r"^theorem\s+([A-Za-z0-9_'.]+)", src, re.MULTILINE)
    n_theorem = len(theorem_names)
    axiom_names = re.findall(r"^axiom\s+([A-Za-z0-9_'.]+)", src, re.MULTILINE)
    n_axiom = len(axiom_names)

    results = {
        "lean_check_cold_wall_s": wall,
        "lean_check_warm_wall_s": wall_warm,
        "proof_file": PROOF_FILE,
        "proof_file_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "proof_file_bytes": len(src.encode("utf-8")),
        "n_theorems_proved": n_theorem,
        "n_axioms_explicit": n_axiom,
        "theorem_names": theorem_names,
        "expected_theorems": EXPECTED_THEOREMS,
        "theorem_inventory_matches": theorem_names == EXPECTED_THEOREMS,
        "axiom_names": axiom_names,
        "expected_axioms": EXPECTED_AXIOMS,
        "axiom_inventory_matches": axiom_names == EXPECTED_AXIOMS,
        "proof_inventory_matches": theorem_names == EXPECTED_THEOREMS
        and axiom_names == EXPECTED_AXIOMS,
        "cold_exit_code": proc.returncode,
        "warm_exit_code": proc2.returncode,
    }

    print(f"\n{'Metric':<36} {'Value':>16}")
    print("-" * 54)
    print(f"{'lean cold wall':<36} {wall:>14.3f} s")
    print(f"{'lean warm wall':<36} {wall_warm:>14.3f} s")
    print(f"{'theorems proved':<36} {n_theorem:>16}")
    print(f"{'axioms (Mathlib roadmap)':<36} {n_axiom:>16}")
    print(f"{'theorem inventory matches':<36} {str(theorem_names == EXPECTED_THEOREMS):>16}")
    print(f"{'axiom inventory matches':<36} {str(axiom_names == EXPECTED_AXIOMS):>16}")
    print(
        f"{'proof inventory matches':<36} "
        f"{str(theorem_names == EXPECTED_THEOREMS and axiom_names == EXPECTED_AXIOMS):>16}"
    )

    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_formal.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {out_path}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
