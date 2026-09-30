# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — locate the installed Julia runtimes native-parity tests run on

"""Locate an installed Julia runtime by minor version for native-parity tests.

The Julia parity tests run the same programs on two runtimes, 1.11 and the
current release. They used to find them only through Juliaup: ``julia
+<channel>`` or a glob under ``~/.julia/juliaup``. Hosted CI installs Julia
through ``setup-julia`` without Juliaup, so every one of them failed there
(``SystemError: opening file "+1.11"``) while passing on a workstation.

A runtime is now taken, in order, from ``SC_NEUROCORE_TEST_JULIA_<KEY>``
(``1_11``, ``1_13``, ``RELEASE``; CI sets these to the executables it
installed), from the Juliaup install directory, or from ``julia +<runtime>``.
Nothing is skipped: a missing runtime fails the test with the variable to set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["julia_runtime", "julia_runtime_variable", "require_julia_runtime"]


def julia_runtime_variable(runtime: str) -> str:
    """Name the environment variable that pins ``runtime``'s executable."""
    return "SC_NEUROCORE_TEST_JULIA_" + runtime.upper().replace(".", "_")


def julia_runtime(runtime: str) -> Path | None:
    """Return the ``julia`` executable of ``runtime`` ("1.11", "1.13" or "release")."""
    configured = os.environ.get(julia_runtime_variable(runtime))
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    if runtime != "release":
        installed = sorted((Path.home() / ".julia/juliaup").glob(f"julia-{runtime}.*/bin/julia"))
        if installed:
            return installed[-1]
    try:
        resolved = subprocess.check_output(
            [
                "julia",
                f"+{runtime}",
                "--startup-file=no",
                "-e",
                "print(joinpath(Sys.BINDIR, Base.julia_exename()))",
            ],
            text=True,
            timeout=20,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(resolved).resolve()
    return path if path.is_file() else None


def require_julia_runtime(runtime: str) -> Path:
    """Return ``runtime``'s executable, failing with how to provide it."""
    path = julia_runtime(runtime)
    assert path is not None, (
        f"Julia {runtime} native runtime required: set {julia_runtime_variable(runtime)} "
        "or install it with Juliaup"
    )
    return path
