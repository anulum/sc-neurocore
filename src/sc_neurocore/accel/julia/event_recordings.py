# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia event-recording runtime bridge

"""Borrow NumPy buffers for synchronous Julia decoding in an operator-owned runtime."""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def julia_recording_enabled() -> bool:
    """Return the explicit Julia opt-in; refuse malformed operator settings."""
    enabled = os.environ.get("SC_NEUROCORE_DATASET_JULIA_ENABLED", "0")
    if enabled not in {"0", "1"}:
        raise RuntimeError("Julia dataset opt-in must be 0 or 1")
    return enabled == "1"


def _configured_runtime() -> tuple[str, str]:
    """Validate operator paths and the single-thread signal contract before each call."""
    executable = Path(os.environ.get("PYTHON_JULIACALL_EXE", ""))
    project = Path(os.environ.get("PYTHON_JULIACALL_PROJECT", ""))
    if not executable.is_absolute() or not executable.is_file():
        raise RuntimeError("Julia dataset runtime requires an existing absolute Julia executable")
    if not project.is_absolute() or not (project / "Project.toml").is_file():
        raise RuntimeError("Julia dataset runtime requires an existing absolute Julia project")
    if os.environ.get("PYTHON_JULIACALL_THREADS") != "1":
        raise RuntimeError("Julia dataset runtime requires one configured thread")
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") not in {"yes", "no"}:
        raise RuntimeError("Julia dataset runtime requires explicit signal handling")
    return str(executable.resolve()), str(project.resolve())


@functools.lru_cache(maxsize=1)
def _recording_module(executable: str, project: str) -> Any:
    """Load the kernel only into the same already-configured JuliaCall runtime."""
    try:
        runtime = importlib.import_module("juliacall")
        if Path(runtime.CONFIG["exepath"]).resolve() != Path(executable) or Path(
            runtime.CONFIG["project"]
        ).resolve() != Path(project):
            raise RuntimeError("JuliaCall runtime configuration changed after startup")
        module = runtime.newmodule("SCNeuroCoreEventRecordings")
        runtime.Main.Base.include(module, str(Path(__file__).parent / "datasets/nmnist.jl"))
    except Exception as error:
        raise RuntimeError("Julia event-recording runtime is unavailable") from error
    return module.NMNISTRecordings


def decode_julia_recording(raw: bytes) -> npt.NDArray[np.float64]:
    """Decode live immutable input into an exclusive float64 destination.

    Parameters
    ----------
    raw : bytes
        Complete N-MNIST records. Runtime paths and opt-in are configured before
        process startup; importing never resolves or installs Julia dependencies.

    Returns
    -------
    numpy.ndarray
        Row-major x, y, polarity and millisecond timestamp columns.

    Raises
    ------
    RuntimeError
        The explicit opt-in, runtime configuration or native call is refused.
    """
    if not julia_recording_enabled():
        raise RuntimeError("Julia event-recording decoder requires explicit opt-in")
    module = _recording_module(*_configured_runtime())
    source = np.frombuffer(raw, dtype=np.uint8)
    output = np.empty((len(raw) // 5, 4), dtype=np.float64)
    code = module.decode_nmnist_pointer(
        source.ctypes.data, source.size, output.ctypes.data, output.size
    )
    if code != 0:
        raise RuntimeError("Julia event-recording decoder refused the recording")
    return output
