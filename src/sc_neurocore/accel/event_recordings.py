# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native event-recording decoding

"""Decode published camera records through native libraries or the NumPy floor."""

from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_selection import select_backend_order
from sc_neurocore.accel.julia.event_recordings import (
    decode_julia_recording,
    julia_recording_enabled,
)

_NATIVE_PATHS = {
    "rust": ("SC_NEUROCORE_DATASET_RUST_LIBRARY", "rust/safety/libnmnist.so"),
    "mojo": ("SC_NEUROCORE_DATASET_MOJO_LIBRARY", "mojo/kernels/libnmnist.so"),
    "go": ("SC_NEUROCORE_DATASET_GO_LIBRARY", "go/services/loaders/libloaders.so"),
}


@functools.lru_cache(maxsize=4)
def _load_native_library(path: str, backend: str) -> ctypes.CDLL:
    """Bind the N-MNIST C decoder from one operator-selected native library."""
    try:
        library = ctypes.CDLL(path)
        decode = library.nmnist_decode_c
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"{backend} event-recording decoder is unavailable") from error
    decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    decode.restype = ctypes.c_int
    return library


def _native_library_path(backend: str) -> Path | None:
    """Locate an explicitly configured library or the installed default artifact."""
    variable, installed_path = _NATIVE_PATHS[backend]
    configured = os.environ.get(variable)
    if configured is not None:
        path = Path(configured)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"{backend} dataset library must be an existing absolute file")
        return path
    path = Path(__file__).parent / installed_path
    return path if path.is_file() else None


def decode_nmnist_recording(
    raw: bytes, *, backend: Literal["auto", "numpy", "rust", "mojo", "julia", "go"] = "auto"
) -> npt.NDArray[np.float64]:
    """Decode N-MNIST bytes with identical address, polarity and timestamp semantics.

    Parameters
    ----------
    raw : bytes
        Complete 40-bit records from one published-format recording.
    backend : {"auto", "numpy", "rust", "mojo", "julia", "go"}
        ``auto`` chooses available native decoders using the host's recorded
        benchmark order, then the NumPy floor. Without measurements Rust
        precedes Mojo, Julia and Go. Explicit native selection refuses missing libraries.
        Operator settings ``SC_NEUROCORE_DATASET_RUST_LIBRARY`` ,
        ``SC_NEUROCORE_DATASET_MOJO_LIBRARY`` and
        ``SC_NEUROCORE_DATASET_GO_LIBRARY`` select absolute library paths;
        Julia requires an explicit operator opt-in and preconfigured JuliaCall
        runtime. Loading never installs dependencies or downloads artifacts.

    Returns
    -------
    numpy.ndarray
        Float64 columns ``x, y, polarity, timestamp_ms``. Microseconds are
        divided by 1000 without encoder-dependent scaling or float32 narrowing.

    Raises
    ------
    ValueError
        Records are incomplete or the backend name is unknown.
    RuntimeError
        A requested or configured native decoder is unavailable or refuses
        the input. Native failure never silently substitutes another backend.
    """
    if backend not in ("auto", "numpy", "rust", "mojo", "julia", "go"):
        raise ValueError("unsupported event-recording backend")
    if len(raw) % 5:
        raise ValueError("N-MNIST file contains an incomplete 40-bit event")
    order = (
        select_backend_order("nmnist-recording", static=("rust", "mojo", "julia", "go", "numpy"))
        if backend == "auto"
        else (backend,)
    )
    for selected in order:
        if selected == "numpy":
            break
        if selected == "julia":
            if backend == "julia" or julia_recording_enabled():
                return decode_julia_recording(raw)
            continue
        path = _native_library_path(selected)
        if path is not None:
            library = _load_native_library(str(path), selected)
            input_array = np.frombuffer(raw, dtype=np.uint8)
            output = np.empty((len(raw) // 5, 4), dtype=np.float64)
            code = library.nmnist_decode_c(
                input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                input_array.size,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.size,
            )
            if code != 0:
                raise RuntimeError(f"{selected} event-recording decoder refused the recording")
            return output
        if backend != "auto":
            raise RuntimeError(f"{selected} event-recording decoder is unavailable")
    events = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 5).astype(np.uint32)
    time_us = ((events[:, 2] & 0x7F) << 16) | (events[:, 3] << 8) | events[:, 4]
    return np.column_stack(
        (events[:, 0], events[:, 1], events[:, 2] >> 7, time_us.astype(np.float64) / 1000.0)
    )
