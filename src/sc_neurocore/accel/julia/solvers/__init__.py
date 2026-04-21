# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continuous ODE Solvers using DiffEq.jl

import shutil
import subprocess
from pathlib import Path


class JuliaFusionSolver:
    """Orchestrates Julia ODE solvers executing the `fusion_solver.jl` logic natively."""

    def __init__(self) -> None:
        self.julia_script = Path(__file__).parent / "fusion_solver.jl"
        julia_bin = shutil.which("julia")
        if not julia_bin:
            raise FileNotFoundError("Julia binary not found on system PATH. Please install Julia.")
        self._julia_bin: str = julia_bin

    def run_dynamics(self, steps: int) -> None:
        """Invoke continuous ODE simulation on the Julia baseline bounds."""
        try:
            print("[Julia Solvers] Executing continuous physics wrapper via DiffEq.jl")
            subprocess.run([self._julia_bin, str(self.julia_script), str(steps)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Julia Solvers] Process execution failed natively: {e}")
            raise
