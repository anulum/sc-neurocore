# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SHD recording readers refuse instead of substituting

"""Refusals of the SHD recording readers, each raised by a real cause.

Operator settings that do not name a usable library or command, a reader
that answers outside the result protocol, and a recording whose vectors are
not numeric must each stop the read. The readers here are real operator
commands written by the test, started and reaped by the production code.
"""

from __future__ import annotations

import os
import signal
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from sc_neurocore.accel.shd_recordings import read_shd_recording
from tests.test_accel_go_shd_abi import recording_file as recording_file

_VARIABLES = (
    "SC_NEUROCORE_SHD_RUST_LIBRARY",
    "SC_NEUROCORE_SHD_GO_LIBRARY",
    "SC_NEUROCORE_SHD_JULIA_EXE",
    "SC_NEUROCORE_SHD_MOJO_EXE",
    "SC_NEUROCORE_SHD_JULIA_HDF5_LIBRARY",
    "SC_NEUROCORE_SHD_MOJO_HDF5_LIBRARY",
)


@pytest.fixture(autouse=True)
def no_operator_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case with no SHD reader selected by the environment."""
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _command(tmp_path: Path, body: str) -> Path:
    """Write a real executable operator command with the given shell body."""
    path = tmp_path / "shd-reader"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


class _Interrupted(Exception):
    """Raised by the timer's handler in the waiting parent."""


@pytest.fixture
def interval_interrupt() -> Iterator[None]:
    """Deliver one real SIGALRM shortly after the read starts, then restore the handler."""

    def interrupt(_number: int, _frame: object) -> None:
        raise _Interrupted

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def test_an_unknown_backend_and_a_path_with_a_nul_are_refused(recording_file: Path) -> None:
    """Neither a backend outside the published set nor a NUL in the path starts a read."""
    with pytest.raises(ValueError, match="invalid SHD backend or recording path"):
        read_shd_recording(recording_file, 0, backend=cast(Any, "fortran"))
    with pytest.raises(ValueError, match="invalid SHD backend or recording path"):
        read_shd_recording(Path(str(recording_file) + "\x00"), 0, backend="numpy")


def test_a_configured_library_that_does_not_exist_is_refused(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library setting must name an existing absolute file."""
    monkeypatch.setenv("SC_NEUROCORE_SHD_RUST_LIBRARY", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="Rust SHD library must be an existing absolute file"):
        read_shd_recording(recording_file, 0, backend="rust")


@pytest.mark.parametrize("backend", ["rust", "julia"])
def test_an_explicit_reader_that_is_not_configured_is_unavailable(
    recording_file: Path, backend: str
) -> None:
    """Asking for a reader that is neither configured nor installed is refused."""
    with pytest.raises(RuntimeError, match=f"{backend.title()} SHD reader is unavailable"):
        read_shd_recording(recording_file, 0, backend=cast(Any, backend))


def test_a_configured_command_that_cannot_be_executed_is_refused(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command setting must name an existing absolute executable file."""
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\n", encoding="utf-8")
    plain.chmod(0o600)
    monkeypatch.setenv("SC_NEUROCORE_SHD_MOJO_EXE", str(plain))
    with pytest.raises(RuntimeError, match="Mojo SHD executable must be an existing absolute"):
        read_shd_recording(recording_file, 0, backend="mojo")


def test_a_configured_hdf5_library_that_does_not_exist_is_refused(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HDF5 library setting must name an existing absolute file."""
    monkeypatch.setenv("SC_NEUROCORE_SHD_MOJO_EXE", str(_command(tmp_path, "exit 0\n")))
    monkeypatch.setenv("SC_NEUROCORE_SHD_MOJO_HDF5_LIBRARY", str(tmp_path / "absent-hdf5.so"))
    with pytest.raises(RuntimeError, match="Mojo HDF5 library must be an existing absolute file"):
        read_shd_recording(recording_file, 0, backend="mojo")


def test_a_reader_that_answers_with_too_few_bytes_is_refused(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful exit with less than a header is an incomplete response."""
    monkeypatch.setenv("SC_NEUROCORE_SHD_MOJO_EXE", str(_command(tmp_path, "printf 'SHD'\n")))
    with pytest.raises(RuntimeError, match="Mojo SHD reader returned an incomplete response"):
        read_shd_recording(recording_file, 0, backend="mojo")


@pytest.mark.parametrize(
    "header",
    [
        struct.pack("<4sQq", b"NOPE", 0, 3),
        struct.pack("<4sQq", b"SHD1", 3, 3),
        struct.pack("<4sQq", b"SHD1", 4, 3),
    ],
    ids=["magic", "count-not-rows", "count-without-values"],
)
def test_a_reader_that_answers_outside_the_protocol_is_refused(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: bytes
) -> None:
    """A wrong magic, a count that is not whole rows, or missing values is an invalid response."""
    answer = tmp_path / "answer.bin"
    answer.write_bytes(header)
    monkeypatch.setenv("SC_NEUROCORE_SHD_MOJO_EXE", str(_command(tmp_path, f"cat {answer}\n")))
    with pytest.raises(RuntimeError, match="Mojo SHD reader returned an invalid response"):
        read_shd_recording(recording_file, 0, backend="mojo")


def test_an_interrupted_read_kills_and_reaps_its_reader(
    recording_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interval_interrupt: None
) -> None:
    """The reader is gone when an interruption of the waiting parent reaches the caller."""
    identity = tmp_path / "reader.pid"
    monkeypatch.setenv(
        "SC_NEUROCORE_SHD_MOJO_EXE",
        str(_command(tmp_path, f"echo $$ > {identity}\nexec sleep 60\n")),
    )
    with pytest.raises(_Interrupted):
        read_shd_recording(recording_file, 0, backend="mojo")
    deadline = time.monotonic() + 5.0
    while not identity.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(int(identity.read_text()), 0)


def test_vectors_that_are_not_numeric_are_refused_by_the_numpy_reader(tmp_path: Path) -> None:
    """A recording whose event vectors hold text is not an SHD recording."""
    path = tmp_path / "text-vectors.h5"
    with h5py.File(path, "w") as handle:
        text = h5py.vlen_dtype(np.dtype("S1"))
        handle.create_dataset("spikes/times", (1,), dtype=text)
        handle.create_dataset("spikes/units", (1,), dtype=text)
        handle["labels"] = np.array([3], dtype=np.uint8)
    with pytest.raises(ValueError, match="SHD events must use numeric variable-length vectors"):
        read_shd_recording(path, 0, backend="numpy")
