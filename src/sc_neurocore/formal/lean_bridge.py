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


class FormalProofEngine:
    """Invokes Lean 4 `safety_bounds.lean` to formally verify mathematical parameters dynamically."""

    def __init__(self):
        self._lean_bin = shutil.which("lean")
        self.proof_file = Path(__file__).parent / "proofs" / "safety_bounds.lean"

    def is_available(self) -> bool:
        return self._lean_bin is not None and self.proof_file.exists()

    def check_proofs(self) -> bool:
        """Invokes the native `lean --check` verification boundary safely testing axioms."""
        if not self.is_available():
            print("[Formal] Lean 4 unavailable on the system path.")
            return False

        print("[Formal] Running formal checking across physical stochastic theorems...")
        try:
            result = subprocess.run([self._lean_bin, str(self.proof_file)], capture_output=True, text=True, check=True)
            if "error" in result.stdout.lower() or "error" in result.stderr.lower():
                print(f"[Formal] Failure natively detected: {result.stderr}")
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"[Formal] Process exception: {e.stderr}")
            return False
