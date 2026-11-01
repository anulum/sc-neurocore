# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Container compiler parallelism contracts

"""Execute the Dockerfile's shell setup with inherited and explicit compiler limits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("build_jobs", "inherited", "expected"),
    [
        (None, (None, None), [None, None]),
        (None, ("4", "6"), ["4", "6"]),
        ("2", ("4", "6"), ["2", "2"]),
    ],
)
def test_container_build_jobs_preserve_defaults_or_forward_explicit_limits(
    build_jobs: str | None,
    inherited: tuple[str | None, str | None],
    expected: list[str | None],
) -> None:
    """Run the production shell setup and inspect its child process environment.

    Parameters
    ----------
    build_jobs : str or None
        Optional Docker build argument; absence leaves compiler defaults intact.
    inherited : tuple of str or None
        Cargo and CMake limits inherited before the Dockerfile's setup runs.
    expected : list of str or None
        Exact limits visible to the next command in the native build step.
    """
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "deploy/Dockerfile").read_text(encoding="utf-8")
    assert "ARG BUILD_JOBS\n" in dockerfile
    instructions = dockerfile.replace("\\\n", "").splitlines()
    build = next(
        line[4:] for line in instructions if line.startswith("RUN ") and "maturin build" in line
    )
    setup, command = build.split("&& cd bridge &&", 1)
    assert command.lstrip().startswith("maturin build --release --locked --interpreter python3")
    environment = os.environ.copy()
    keys = ("CARGO_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL")
    for key in ("BUILD_JOBS", *keys):
        environment.pop(key, None)
    for key, value in zip(keys, inherited):
        if value is not None:
            environment[key] = value
    if build_jobs is not None:
        environment["BUILD_JOBS"] = build_jobs
    shell = shutil.which("sh")
    assert shell is not None, "Container build contracts require a POSIX shell."
    probe = "import json,os; print(json.dumps([os.environ.get(k) for k in " + repr(keys) + "]))"
    script = setup + "\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(probe)
    result = subprocess.run(
        [shell, "-ec", script], env=environment, capture_output=True, text=True, timeout=20
    )
    result.check_returncode()
    assert json.loads(result.stdout) == expected
