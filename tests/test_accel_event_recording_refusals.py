# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — N-MNIST recording decoders refuse instead of substituting

"""Refusals of the N-MNIST recording decoders, each raised by a real cause.

A native decoder that cannot be loaded, that is not installed, or that reports
failure must stop the read; none of these may turn into a NumPy result. The
libraries here are real files: one that is not a library, and one compiled by
the test whose decoder returns a failure code.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from sc_neurocore.accel.event_recordings import decode_nmnist_recording
from sc_neurocore.accel.julia.event_recordings import decode_julia_recording

_REFUSING_DECODER = """
#include <stddef.h>
#include <stdint.h>

int nmnist_decode_c(const uint8_t *input, size_t input_size, double *output, size_t output_size) {
    (void)input; (void)input_size; (void)output; (void)output_size;
    return 7;
}
"""


@pytest.fixture(scope="module")
def refusing_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile a real shared library whose decoder always reports failure."""
    directory = tmp_path_factory.mktemp("refusing-decoder")
    source = directory / "refusing.c"
    source.write_text(_REFUSING_DECODER, encoding="utf-8")
    output = directory / "librefusing.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", str(output), str(source)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return output


def test_an_unknown_backend_name_is_refused() -> None:
    """A backend outside the published set is refused before any byte is read."""
    with pytest.raises(ValueError, match="unsupported event-recording backend"):
        decode_nmnist_recording(bytes(5), backend=cast(Any, "fortran"))


def test_a_configured_file_that_is_not_a_library_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing file the loader cannot open stops the read; NumPy is not substituted."""
    broken = tmp_path / "broken.so"
    broken.write_bytes(b"not an ELF library")
    monkeypatch.setenv("SC_NEUROCORE_DATASET_RUST_LIBRARY", str(broken))
    with pytest.raises(RuntimeError, match="rust event-recording decoder is unavailable"):
        decode_nmnist_recording(bytes(5), backend="rust")


def test_a_decoder_that_reports_failure_stops_the_read(
    refusing_library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native failure code is a refusal, not an empty or partial recording."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_GO_LIBRARY", str(refusing_library))
    with pytest.raises(RuntimeError, match="go event-recording decoder refused the recording"):
        decode_nmnist_recording(bytes(5), backend="go")


def test_an_explicit_backend_with_no_library_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for a native decoder that is neither configured nor installed is refused."""
    monkeypatch.delenv("SC_NEUROCORE_DATASET_MOJO_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="mojo event-recording decoder is unavailable"):
        decode_nmnist_recording(bytes(5), backend="mojo")


@pytest.mark.parametrize(
    ("variable", "value", "reason"),
    [
        ("PYTHON_JULIACALL_THREADS", "2", "one configured thread"),
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "sometimes", "explicit signal handling"),
    ],
)
def test_julia_refuses_a_runtime_outside_its_contract(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, reason: str
) -> None:
    """A thread count or signal setting outside the contract is refused before any call."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
    monkeypatch.setenv("PYTHON_JULIACALL_THREADS", "1")
    monkeypatch.setenv("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError, match=reason):
        decode_julia_recording(bytes(5))


def test_julia_refuses_a_project_other_than_the_running_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid project that is not the one the runtime started with is refused."""
    other = tmp_path / "other-project"
    other.mkdir()
    (other / "Project.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
    monkeypatch.setenv("PYTHON_JULIACALL_THREADS", "1")
    monkeypatch.setenv("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")
    monkeypatch.setenv("PYTHON_JULIACALL_PROJECT", str(other))
    assert Path(os.environ["PYTHON_JULIACALL_EXE"]).is_file()
    with pytest.raises(RuntimeError, match="Julia event-recording runtime is unavailable"):
        decode_julia_recording(bytes(5))


def test_julia_decoder_refuses_an_incomplete_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Julia kernel itself refuses bytes that do not form whole 40-bit records."""
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
    with pytest.raises(RuntimeError, match="Julia event-recording decoder refused the recording"):
        decode_julia_recording(bytes(7))
