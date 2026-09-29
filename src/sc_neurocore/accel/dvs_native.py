# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Converted DVS recording format contract

"""Run configured native DVS recording commands with explicit ownership and refusal."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

_HEADER = struct.Struct("<4sQ")


def native_executable(backend: Literal["go", "rust", "julia", "mojo"]) -> Path | None:
    """Validate the selected operator command without discovery or runtime installation."""
    label = {"go": "Go", "rust": "Rust", "julia": "Julia", "mojo": "Mojo"}[backend]
    configured = os.environ.get(f"SC_NEUROCORE_DVS_{backend.upper()}_EXE")
    if configured is None:
        return None
    executable = Path(configured)
    if sys.platform != "linux":
        raise RuntimeError(f"guarded {label} DVS reading requires Linux")
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(f"{label} DVS executable must be an existing absolute executable path")
    return executable


def read_native_recording(
    path: Path, maximum: int, executable: Path, *, backend: Literal["go", "rust", "julia", "mojo"]
) -> npt.NDArray[np.float64]:
    """Read an owned native event matrix or refuse the entire command response.

    Parameters
    ----------
    path : pathlib.Path
        A local converted NPY camera recording.
    maximum : int
        Validated returned float64 matrix byte limit.
    executable : pathlib.Path
        Validated operator-owned Linux DVS command.
    backend : {"go", "rust", "julia", "mojo"}
        Selected implementation used in refusal diagnostics. Julia takes an installed
        runtime executable and runs the packaged DVS script without startup files.

    Returns
    -------
    numpy.ndarray
        Writable owned C-contiguous four-column float64 events.

    Raises
    ------
    RuntimeError
        Native refusal, timeout, incomplete frame, invalid shape or oversized result.
    OSError
        The declared executable cannot be started.

    Notes
    -----
    The child inherits the compute process group, receives this parent's PID
    and has its own 30-second guard before loading the recording reader. The parent
    bounds startup and communicate together to 30 seconds, kills
    and reaps on interruption. Input/native buffers and pipe copies consume
    additional memory; the result budget is not an aggregate memory limit.
    """
    label = {"go": "Go", "rust": "Rust", "julia": "Julia", "mojo": "Mojo"}[backend]
    command = [str(executable)]
    if backend == "julia":
        command.extend(
            [
                "--startup-file=no",
                "--check-bounds=yes",
                "--depwarn=error",
                "--threads=1",
                str(Path(__file__).parent / "julia/datasets/dvs_cli.jl"),
            ]
        )
    command.extend([str(path), str(maximum), str(os.getpid())])
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise RuntimeError(f"{label} DVS reader exceeded its 30 second lifetime") from error
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"{label} DVS reader failed or refused the recording")
    if len(output) < _HEADER.size:
        raise RuntimeError(f"{label} DVS reader returned an incomplete response")
    magic, count = _HEADER.unpack_from(output)
    if (
        magic != b"DVS1"
        or count % 4
        or count > maximum // 8
        or len(output) != _HEADER.size + count * 8
    ):
        raise RuntimeError(f"{label} DVS reader returned an invalid response")
    return np.array(
        np.frombuffer(output, dtype="<f8", offset=_HEADER.size).reshape((-1, 4)),
        dtype=np.float64,
        order="C",
        copy=True,
    )


def go_executable() -> Path | None:
    """Validate the explicitly declared Go DVS command for existing callers."""
    return native_executable("go")


def read_go_recording(path: Path, maximum: int, executable: Path) -> npt.NDArray[np.float64]:
    """Read the declared Go command through the shared guarded binary protocol."""
    return read_native_recording(path, maximum, executable, backend="go")
