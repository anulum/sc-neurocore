# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Continuous ODE Solvers using DiffEq.jl

"""Run the reference ``fusion_solver.jl`` ODE script through a Julia executable.

The run and its failure are reported through the module logger, not printed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

_LOG = logging.getLogger(__name__)


class JuliaFusionSolver:
    """Run ``fusion_solver.jl`` (DifferentialEquations.jl) in a Julia subprocess.

    Parameters
    ----------
    julia_bin : str, optional
        Julia executable to run; the ``julia`` found on ``PATH`` when omitted.

    Raises
    ------
    FileNotFoundError
        No executable was given and ``PATH`` holds no ``julia``.
    """

    def __init__(self, julia_bin: str | None = None) -> None:
        """Locate the script and the Julia executable."""
        self.julia_script = Path(__file__).parent / "fusion_solver.jl"
        found = shutil.which("julia") if julia_bin is None else julia_bin
        if not found:
            raise FileNotFoundError("Julia binary not found on system PATH. Please install Julia.")
        self._julia_bin: str = found

    def run_dynamics(self, steps: int) -> None:
        """Run the script for ``steps`` steps and wait for it to finish.

        Parameters
        ----------
        steps : int
            Step count passed to the script as its only argument.

        Raises
        ------
        subprocess.CalledProcessError
            The Julia process exits with a non-zero status; the failure is
            logged before it is raised again.
        """
        _LOG.info("Running %s with %d steps", self.julia_script.name, steps)
        try:
            subprocess.run([self._julia_bin, str(self.julia_script), str(steps)], check=True)
        except subprocess.CalledProcessError as e:
            _LOG.error("Julia fusion solver failed: %s", e)
            raise
