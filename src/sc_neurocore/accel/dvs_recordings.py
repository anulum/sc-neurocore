# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Converted DVS recording format contract

"""Read one bounded real numeric NPY camera recording without pickle or coercion."""

from __future__ import annotations

import ast
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.dvs_native import (
    go_executable,
    native_executable,
    read_go_recording,
    read_native_recording,
)

DEFAULT_DVS_EVENT_BYTES = 64 * 1024 * 1024
_MAX_HEADER_BYTES = 10_000


def _exact(stream: BinaryIO, size: int) -> bytes:
    """Read exactly the declared bytes or refuse an incomplete recording."""
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("incomplete DVS NPY recording")
    return value


def _header(stream: BinaryIO) -> tuple[int, np.dtype[np.generic], bool]:
    """Validate an unambiguous rank-two real numeric NPY header before payload allocation."""
    if _exact(stream, 6) != b"\x93NUMPY":
        raise ValueError("DVS recording must be a single NPY array")
    version = tuple(_exact(stream, 2))
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("unsupported DVS NPY format version")
    width = 2 if version == (1, 0) else 4
    length = struct.unpack("<H" if width == 2 else "<I", _exact(stream, width))[0]
    if not 0 < length <= _MAX_HEADER_BYTES:
        raise ValueError("DVS NPY header exceeds its byte limit")
    raw = _exact(stream, length)
    if not raw.endswith(b"\n"):
        raise ValueError("DVS NPY header must end with a newline")
    try:
        expression = ast.parse(raw.decode("utf8" if version == (3, 0) else "latin1"), mode="eval")
        if not isinstance(expression.body, ast.Dict):
            raise ValueError("DVS NPY header must be a dictionary")
        keys = [ast.literal_eval(key) for key in expression.body.keys if key is not None]
        if len(keys) != 3 or set(keys) != {"descr", "fortran_order", "shape"}:
            raise ValueError("DVS NPY header fields must be unique and complete")
        metadata = ast.literal_eval(expression)
    except (SyntaxError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid DVS NPY header") from error
    shape, order, descriptor = metadata["shape"], metadata["fortran_order"], metadata["descr"]
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) or part < 0 for part in shape)
        or shape[1] != 4
    ):
        raise ValueError("camera events must have four columns")
    if not isinstance(order, bool) or not isinstance(descriptor, str):
        raise ValueError("invalid DVS NPY layout or datatype declaration")
    try:
        dtype = np.dtype(descriptor)
    except TypeError as error:
        raise ValueError("invalid DVS NPY datatype") from error
    if dtype.kind not in "buif":
        raise ValueError("camera events must use real numeric scalar values")
    return shape[0], dtype, order


def read_dvs_recording(
    path: Path,
    *,
    maximum_bytes: int = DEFAULT_DVS_EVENT_BYTES,
    backend: Literal["auto", "numpy", "go", "rust", "julia", "mojo"] = "auto",
) -> npt.NDArray[np.float64]:
    """Read a converted camera recording as an owned row-major event matrix.

    Parameters
    ----------
    path : pathlib.Path
        One NPY array with x, y, polarity and millisecond timestamp columns.
        Versions 1.0, 2.0 and 3.0, either byte order, and C/Fortran layouts
        are supported. Real integer, Boolean and floating scalar types are
        converted to float64; extended precision follows NumPy conversion.
    maximum_bytes : int
        Returned matrix byte limit, default 64 MiB. Header parsing uses at
        most 10,000 bytes. Input bytes and temporary copies are additional
        memory; this is not an aggregate memory cap.

    backend : {"auto", "numpy", "go", "rust", "julia", "mojo"}
        Auto selects one explicitly declared SC_NEUROCORE_DVS_GO_EXE or
        SC_NEUROCORE_DVS_RUST_EXE, SC_NEUROCORE_DVS_JULIA_EXE or
        SC_NEUROCORE_DVS_MOJO_EXE, otherwise
        NumPy; multiple declarations refuse. Julia uses the declared runtime
        with the packaged script, one thread and startup files disabled.
        Explicit native selection requires an existing absolute Linux executable.
        A selected native refusal never falls back to NumPy.

    Returns
    -------
    numpy.ndarray
        Writable owned float64 array with shape (N, 4), without time rescaling.

    Raises
    ------
    ValueError
        Invalid budget, header, shape, nonreal dtype, truncation, extra content
        or result exceeding its declared budget. Pickle is never invoked.
    OSError
        The local recording or native executable is unreadable.
    RuntimeError
        A selected native command is unavailable, refuses or violates its bounded protocol.

    Notes
    -----
    Geometry, finite values, time monotonicity and manifest identity are
    checked by the caller's dataset and encoder contracts. No download or
    synthetic substitution occurs. NumPy is the reference implementation. Each selected native backend
    reads the recording directly; no native refusal falls back to NumPy.
    """
    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or not 0 <= maximum_bytes <= sys.maxsize
    ):
        raise ValueError("DVS event budget must be a non-negative native integer")
    if backend not in ("auto", "numpy", "go", "rust", "julia", "mojo"):
        raise ValueError("unsupported DVS recording backend")
    selected: Literal["go", "rust", "julia", "mojo"] = (
        backend if backend in ("rust", "julia", "mojo") else "go"
    )
    if backend == "auto":
        declared = [
            kind
            for kind in ("go", "rust", "julia", "mojo")
            if f"SC_NEUROCORE_DVS_{kind.upper()}_EXE" in os.environ
        ]
        if len(declared) > 1:
            raise RuntimeError("automatic DVS selection requires one native declaration")
        if "rust" in declared:
            selected = "rust"
        elif "julia" in declared:
            selected = "julia"
        elif "mojo" in declared:
            selected = "mojo"
    executable = (
        None
        if backend == "numpy"
        else (go_executable() if selected == "go" else native_executable(selected))
    )
    if executable is not None:
        if selected == "go":
            return read_go_recording(path, maximum_bytes, executable)
        return read_native_recording(path, maximum_bytes, executable, backend=selected)
    if backend in ("go", "rust", "julia", "mojo"):
        label = {"go": "Go", "rust": "Rust", "julia": "Julia", "mojo": "Mojo"}[backend]
        raise RuntimeError(f"{label} DVS backend requires SC_NEUROCORE_DVS_{backend.upper()}_EXE")
    with path.open("rb") as stream:
        count, dtype, fortran = _header(stream)
        if count > maximum_bytes // 32:
            raise ValueError("DVS recording exceeds the event budget")
        payload = _exact(stream, count * 4 * dtype.itemsize)
        if stream.read(1):
            raise ValueError("DVS recording contains extra content after its array")
    values = np.frombuffer(payload, dtype=dtype).reshape((count, 4), order="F" if fortran else "C")
    return np.array(values, dtype=np.float64, order="C", copy=True)
