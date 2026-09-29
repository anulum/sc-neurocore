# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native event-recording decoding

"""Select an indexed SHD reader without mixing Julia and system HDF5 libraries."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_selection import select_backend_order

DEFAULT_SHD_EVENT_BYTES = 64 * 1024 * 1024
_SHD_HEADER = struct.Struct("<4sQq")


_NATIVE_SHD = {
    "rust": ("SC_NEUROCORE_SHD_RUST_LIBRARY", "rust/safety/libshd.so"),
    "go": ("SC_NEUROCORE_SHD_GO_LIBRARY", "go/services/loaders/libshd.so"),
}


def _native_library(backend: str) -> Path | None:
    """Resolve the selected SHD artifact independently of N-MNIST builds."""
    variable, installed = _NATIVE_SHD[backend]
    configured = os.environ.get(variable)
    if configured is not None:
        path = Path(configured)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"{backend.title()} SHD library must be an existing absolute file")
        return path
    path = Path(__file__).parent / installed
    return path if path.is_file() else None


def _read_native(
    path: Path, index: int, maximum: int, library: Path, backend: str
) -> tuple[npt.NDArray[np.float64], int]:
    """Run and reap the HDF5 process, refusing unsuccessful or malformed responses."""
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).with_name("shd_worker.py")),
        str(library),
        str(path),
        str(index),
        str(maximum),
        str(os.getpid()),
    ]
    return _read_process(command, maximum, backend)


def _read_process(
    command: list[str], maximum: int, backend: str
) -> tuple[npt.NDArray[np.float64], int]:
    """Own a bounded native process and verify its exact binary result before returning."""
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise RuntimeError(
            f"{backend.title()} SHD reader exceeded its 30 second lifetime"
        ) from error
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"{backend.title()} SHD reader failed or refused the recording")
    if len(output) < _SHD_HEADER.size:
        raise RuntimeError(f"{backend.title()} SHD reader returned an incomplete response")
    magic, count, label = _SHD_HEADER.unpack_from(output)
    if (
        magic != b"SHD1"
        or count % 4
        or count > maximum // 8
        or len(output) != _SHD_HEADER.size + count * 8
    ):
        raise RuntimeError(f"{backend.title()} SHD reader returned an invalid response")
    events = np.frombuffer(output, dtype=np.float64, offset=_SHD_HEADER.size).reshape(-1, 4).copy()
    return events, label


def _configured_executable(backend: str) -> Path | None:
    """Validate explicit installed Julia selection without invoking an installer or launcher."""
    configured = os.environ.get(f"SC_NEUROCORE_SHD_{backend.upper()}_EXE")
    if configured is None:
        return None
    executable = Path(configured)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(
            f"{backend.title()} SHD executable must be an existing absolute executable file"
        )
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            f"supervised {backend.title()} SHD reads require Linux parent-death signals"
        )
    return executable


def _hdf5_library(backend: str) -> str:
    """Resolve the operator's explicit native HDF5 library or the system library name."""
    name = f"SC_NEUROCORE_SHD_{backend.upper()}_HDF5_LIBRARY"
    library = os.environ.get(name, "libhdf5_serial.so")
    if name in os.environ and (not Path(library).is_absolute() or not Path(library).is_file()):
        raise RuntimeError(f"{backend.title()} HDF5 library must be an existing absolute file")
    return library


def _read_mojo(
    path: Path, index: int, maximum: int, executable: Path
) -> tuple[npt.NDArray[np.float64], int]:
    """Run the operator's compiled native Mojo CLI through the bounded recording protocol."""
    return _read_process(
        [
            str(executable),
            str(path),
            str(index),
            str(maximum),
            _hdf5_library("mojo"),
            str(os.getpid()),
        ],
        maximum,
        "mojo",
    )


def _read_julia(
    path: Path, index: int, maximum: int, executable: Path
) -> tuple[npt.NDArray[np.float64], int]:
    """Run the real standard-library Julia kernel with the common bounded result protocol."""
    library = _hdf5_library("julia")
    return _read_process(
        [
            str(executable),
            "--startup-file=no",
            "--history-file=no",
            "--threads=1",
            "--check-bounds=yes",
            "--depwarn=error",
            str(Path(__file__).parent / "julia/datasets/shd_cli.jl"),
            str(path),
            str(index),
            str(maximum),
            library,
            str(os.getpid()),
        ],
        maximum,
        "julia",
    )


def _read_numpy(path: Path, index: int, maximum: int) -> tuple[npt.NDArray[np.float64], int]:
    """Read the same numeric variable-length row contract through h5py."""
    import h5py

    with h5py.File(path, "r") as handle:
        times, units, labels = handle["spikes/times"], handle["spikes/units"], handle["labels"]
        if times.ndim != 1 or units.shape != times.shape or labels.shape != times.shape:
            raise ValueError("SHD datasets must have matching recording counts")
        if index >= len(labels):
            raise ValueError("event sample index is outside the recording file")
        if labels.dtype.kind not in "iu":
            raise ValueError("SHD label datatype is incompatible")
        for vector in (times, units):
            base = h5py.check_vlen_dtype(vector.dtype)
            if base is None or np.dtype(base).kind not in "fiu":
                raise ValueError("SHD events must use numeric variable-length vectors")
        time = np.asarray(times[index], dtype=np.float64)
        channels = np.asarray(units[index], dtype=np.float64)
        label = int(labels[index])
    if time.ndim != 1 or channels.shape != time.shape:
        raise ValueError("auditory event times and channels must be matching vectors")
    if time.size > maximum // 32:
        raise ValueError("SHD recording exceeds the event budget")
    if not -(1 << 63) <= label < (1 << 63):
        raise ValueError("SHD label cannot be represented as int64")
    events = np.zeros((time.size, 4), dtype=np.float64)
    events[:, 0] = channels
    events[:, 3] = time * 1000.0
    return events, label


def read_shd_recording(
    path: Path,
    index: int,
    *,
    maximum_bytes: int = DEFAULT_SHD_EVENT_BYTES,
    backend: Literal["auto", "numpy", "rust", "go", "julia", "mojo"] = "auto",
) -> tuple[npt.NDArray[np.float64], int]:
    """Read one auditory recording and its label through a selected real reader.

    Parameters
    ----------
    path : pathlib.Path
        Local HDF5 file; the caller verifies dataset-manifest identity.
    index : int
        Non-negative recording index, identical in all three datasets.
    maximum_bytes : int
        Event-result byte budget, default 64 MiB. Zero admits empty rows.
        HDF5 input vectors, IPC bytes and temporary copies are additional memory.
    backend : {"auto", "numpy", "rust", "go", "julia", "mojo"}
        Auto uses measured ``shd-recording`` order, otherwise Rust, Go, Julia, Mojo then NumPy.
        Rust uses ``SC_NEUROCORE_SHD_RUST_LIBRARY`` or installed
        ``rust/safety/libshd.so``; Go requires an existing
        ``SC_NEUROCORE_SHD_GO_LIBRARY`` absolute path
        or installed ``go/services/loaders/libshd.so``. Its read runs in a
        fresh process to separate Julia and system HDF5 shared libraries.
        Julia requires an existing absolute ``SC_NEUROCORE_SHD_JULIA_EXE``
        executable and system HDF5; its CLI uses only the installed standard
        library. Linux parent-death signals bound its lifetime when the caller
        disappears. Mojo requires an existing compiled Linux
        ``SC_NEUROCORE_SHD_MOJO_EXE`` and can select system HDF5 via
        ``SC_NEUROCORE_SHD_MOJO_HDF5_LIBRARY``. An attempted native read never silently falls back or downloads a file.

    Returns
    -------
    tuple
        Writable float64 ``(events, 4)`` x/y/polarity/millisecond array and label.

    Raises
    ------
    ValueError
        Index, budget, backend or NumPy recording format is invalid.
    OSError
        A NumPy recording is unreadable or native process creation fails.
    RuntimeError
        A selected native library is absent, fails, times out or returns invalid data.

    Notes
    -----
    The operator owns native library code. Each native read lifetime is 30 seconds;
    every worker is reaped. Reads do not certify event geometry or finite values.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index > sys.maxsize:
        raise ValueError("SHD index must be a non-negative native integer")
    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or not 0 <= maximum_bytes <= sys.maxsize
    ):
        raise ValueError("SHD event budget must be a non-negative native integer")
    if backend not in ("auto", "numpy", "rust", "go", "julia", "mojo") or "\x00" in str(path):
        raise ValueError("invalid SHD backend or recording path")
    order = (
        select_backend_order("shd-recording", static=("rust", "go", "julia", "mojo", "numpy"))
        if backend == "auto"
        else (backend,)
    )
    for selected in order:
        if selected == "numpy":
            return _read_numpy(path, index, maximum_bytes)
        if selected in ("julia", "mojo"):
            executable = _configured_executable(selected)
            if executable is not None:
                reader = _read_julia if selected == "julia" else _read_mojo
                return reader(path, index, maximum_bytes, executable)
            if backend == selected:
                raise RuntimeError(f"{selected.title()} SHD reader is unavailable")
        if selected in _NATIVE_SHD:
            library = _native_library(selected)
            if library is not None:
                return _read_native(path, index, maximum_bytes, library, selected)
            if backend != "auto":
                raise RuntimeError(f"{backend.title()} SHD reader is unavailable")
    raise RuntimeError("no supported SHD reader is available")
