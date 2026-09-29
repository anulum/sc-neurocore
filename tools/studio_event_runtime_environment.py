# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit Julia runtime export for CI workers

"""Warm JuliaCall during CI provisioning and retain its exact runtime paths."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path

from sc_neurocore.studio.platform.storage_event_worker_configuration import EventJuliaRuntime


def export_runtime_environment(runtime: EventJuliaRuntime, github_env: Path) -> None:
    """Append installed Julia paths and worker settings to a runner environment file.

    Parameters
    ----------
    runtime : EventJuliaRuntime
        Installed executable, PythonCall project and explicit signal policy.
    github_env : Path
        GitHub runner environment file for subsequent steps.

    Raises
    ------
    ValueError
        A path contains a newline that could inject another environment entry.
    OSError
        The runner environment file cannot be written.
    """
    values = {
        "PYTHON_JULIACALL_EXE": str(runtime.executable),
        "PYTHON_JULIACALL_PROJECT": str(runtime.project),
        "PYTHON_JULIACALL_THREADS": "1",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": runtime.handle_signals,
    }
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ValueError("Julia runtime paths cannot contain environment line separators")
    with github_env.open("a", encoding="utf-8", newline="\n") as output:
        output.write("".join(f"{key}={value}\n" for key, value in values.items()))


def main(argv: Sequence[str] | None = None) -> int:
    """Provision JuliaCall once and export the selected runtime for later CI steps.

    Parameters
    ----------
    argv : sequence of str, optional
        Command arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero after successful runtime initialization and export.

    Notes
    -----
    This command belongs to the provisioning step: JuliaCall may install its
    dependencies here. Isolated workers receive explicit installed paths and
    continue to refuse missing runtimes rather than provision them.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-env", required=True, type=Path)
    args = parser.parse_args(argv)
    julia = importlib.import_module("juliacall")
    julia.Main.seval("1+1")
    runtime = EventJuliaRuntime(
        executable=Path(julia.CONFIG["exepath"]).resolve(strict=True),
        project=Path(julia.CONFIG["project"]).resolve(strict=True),
        handle_signals="no",
    )
    export_runtime_environment(runtime, args.github_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
