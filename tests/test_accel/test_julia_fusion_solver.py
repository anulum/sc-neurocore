# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia fusion solver wrapper tests

"""Run the Julia fusion-solver wrapper against real stand-in Julia executables.

A POSIX shell script takes the place of ``julia``: it records its arguments, or
fails, so the wrapper's subprocess call and its logged run and failure are real.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.accel.julia import solvers
from sc_neurocore.accel.julia.solvers import JuliaFusionSolver

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="a POSIX shell script stands in for the Julia executable"
)


def _stand_in_julia(directory: Path, body: str) -> str:
    """Write an executable ``julia`` stand-in script and return its path.

    Parameters
    ----------
    directory : pathlib.Path
        Directory that receives the script.
    body : str
        Shell commands after the interpreter line.

    Returns
    -------
    str
        Path of the executable script.
    """
    julia = directory / "julia"
    julia.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    julia.chmod(0o755)
    return str(julia)


def _records(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    """Return the wrapper's captured log levels and messages."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == solvers.__name__
    ]


@posix_only
def test_run_dynamics_runs_the_script_with_the_step_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The executable receives the bundled script and the step count, and the run is logged."""
    julia = _stand_in_julia(tmp_path, 'printf "%s\\n" "$@" > "$(dirname "$0")/call.txt"\n')
    solver = JuliaFusionSolver(julia_bin=julia)
    with caplog.at_level(logging.INFO, logger=solvers.__name__):
        solver.run_dynamics(7)
    script, steps = (tmp_path / "call.txt").read_text(encoding="utf-8").splitlines()
    assert Path(script) == solver.julia_script and solver.julia_script.is_file()
    assert steps == "7"
    assert _records(caplog) == [(logging.INFO, "Running fusion_solver.jl with 7 steps")]


@posix_only
def test_a_failed_julia_run_is_logged_and_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-zero exit is logged as an error and raised to the caller."""
    solver = JuliaFusionSolver(julia_bin=_stand_in_julia(tmp_path, "exit 4\n"))
    with (
        caplog.at_level(logging.INFO, logger=solvers.__name__),
        pytest.raises(subprocess.CalledProcessError) as failure,
    ):
        solver.run_dynamics(3)
    assert failure.value.returncode == 4
    (_, (level, message)) = _records(caplog)
    assert level == logging.ERROR and message.startswith("Julia fusion solver failed: ")


def test_without_julia_on_path_construction_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no executable given and none on ``PATH``, construction is refused."""
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Julia binary not found"):
        JuliaFusionSolver()


@pytest.mark.skipif(shutil.which("julia") is None, reason="no julia on PATH")
def test_the_julia_on_path_is_the_default() -> None:
    """Without ``julia_bin`` the solver uses the ``julia`` found on ``PATH``."""
    assert JuliaFusionSolver()._julia_bin == shutil.which("julia")
