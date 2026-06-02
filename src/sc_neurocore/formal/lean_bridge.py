# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal Proof bindings

import subprocess
import shutil
from pathlib import Path

EXPECTED_AXIOMS: tuple[str, ...] = (
    "sc_precision_numerator_bound",
    "sc_add_preserves_range",
)
EXPECTED_THEOREMS: tuple[str, ...] = (
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
)


class FormalProofEngine:
    """Invokes Lean 4 `safety_bounds.lean` to formally verify mathematical parameters dynamically."""

    def __init__(self) -> None:
        self._lean_bin: str | None = shutil.which("lean")
        self.proof_file = Path(__file__).parent / "proofs" / "safety_bounds.lean"

    def is_available(self) -> bool:
        return self._lean_bin is not None and self.proof_file.exists()

    def list_axioms(self) -> list[str]:
        """Return explicit Lean axiom names declared in the bundled proof file."""
        if not self.proof_file.exists():
            return []

        axioms: list[str] = []
        for line in self.proof_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("axiom "):
                parts = line.split()
                if len(parts) >= 2:
                    axioms.append(parts[1])
        return axioms

    def list_theorems(self) -> list[str]:
        """Return explicit top-level Lean theorem names declared in the proof file."""
        if not self.proof_file.exists():
            return []

        theorems: list[str] = []
        for line in self.proof_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("theorem "):
                parts = line.split()
                if len(parts) >= 2:
                    theorems.append(parts[1])
        return theorems

    def axiom_inventory_matches(self) -> bool:
        """Return True only when the proof file contains the reviewed axiom set."""
        return tuple(self.list_axioms()) == EXPECTED_AXIOMS

    def theorem_inventory_matches(self) -> bool:
        """Return True only when the proof file contains the reviewed theorem set."""
        return tuple(self.list_theorems()) == EXPECTED_THEOREMS

    def proof_inventory_matches(self) -> bool:
        """Return True only when reviewed theorem and axiom inventories match."""
        return self.theorem_inventory_matches() and self.axiom_inventory_matches()

    def check_proofs(self) -> bool:
        """Invoke native Lean elaboration for the bundled proof boundary."""
        if not self.is_available():
            print("[Formal] Lean 4 unavailable on the system path.")
            return False

        assert self._lean_bin is not None, "is_available() guarantees a non-None bin path"
        print("[Formal] Running formal checking across physical stochastic theorems...")
        try:
            result = subprocess.run(
                [self._lean_bin, str(self.proof_file)],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            if "error" in result.stdout.lower() or "error" in result.stderr.lower():
                print(f"[Formal] Failure natively detected: {result.stderr}")
                return False
            if not self.proof_inventory_matches():
                print("[Formal] Proof inventory mismatch.")
                print(f"[Formal] Expected theorems: {list(EXPECTED_THEOREMS)}")
                print(f"[Formal] Found theorems: {self.list_theorems()}")
                print(f"[Formal] Expected axioms: {list(EXPECTED_AXIOMS)}")
                print(f"[Formal] Found axioms: {self.list_axioms()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print("[Formal] Lean proof check timed out.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"[Formal] Process exception: {e.stderr}")
            return False
