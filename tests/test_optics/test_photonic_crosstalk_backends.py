# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executable photonic crosstalk backend parity

"""Compile every maintained crosstalk mirror and enforce the Python contract."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.mojo.isa_baseline import pin_isa
from sc_neurocore.optics.photonic_emitter import WaveguidePair

_REPOSITORY = Path(__file__).resolve().parents[2]
_RUST_SOURCE = _REPOSITORY / "src/sc_neurocore/accel/rust/safety/photonic_emitter.rs"
_GO_ROOT = _REPOSITORY / "src/sc_neurocore/accel/go"
_JULIA_TEST = _REPOSITORY / "src/sc_neurocore/accel/julia/photonic_emitter_parity_test.jl"
_MOJO_SOURCE = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/photonic_emitter.mojo"


def _require_tool(name: str) -> str:
    """Return an executable tool path or skip its optional backend contract."""
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"optional photonic backend tool is unavailable: {name}")
    return executable


def _run(command: list[str], *, cwd: Path = _REPOSITORY, timeout: int = 120) -> str:
    """Execute a bounded backend command and return standard output."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def test_polyglot_sources_are_real_bounded_numeric_kernels() -> None:
    """Keep translated Python comments and no-op acceleration claims removed."""
    sources = (
        _RUST_SOURCE,
        _GO_ROOT / "services/photonic_emitter/photonic_emitter.go",
        _JULIA_TEST.parent / "optics/photonic_emitter.jl",
        _MOJO_SOURCE,
    )
    forbidden = (
        "Go-accelerated",
        "return 0  # return",
        "CrosstalkModelState",
        "#![allow(unused_variables",
        "Mock result for testing without Meep",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 350
        assert not any(marker in text for marker in forbidden)


def test_standalone_rust_safety_contract_executes(tmp_path: Path) -> None:
    """Compile and execute golden, ceiling, bank, and rejection Rust tests."""
    binary = tmp_path / "photonic_crosstalk_rust_tests"
    output = _run(
        [
            _require_tool("rustc"),
            "--edition",
            "2021",
            "--test",
            str(_RUST_SOURCE),
            "-o",
            str(binary),
        ]
    )
    assert output == ""
    test_output = _run([str(binary)])
    assert "4 passed; 0 failed" in test_output


def test_go_service_contract_executes() -> None:
    """Execute the Go golden, bank, rejection, and zero-length contracts."""
    output = _run(
        [_require_tool("go"), "test", "./services/photonic_emitter"],
        cwd=_GO_ROOT,
    )
    assert "github.com/anulum/sc-neurocore/accel/services/photonic_emitter" in output


def test_julia_contract_executes() -> None:
    """Execute the Julia pair, bank, batch, and rejection parity suite."""
    output = _run(
        [
            _require_tool("julia"),
            "--startup-file=no",
            "--project=@stdlib",
            str(_JULIA_TEST),
        ]
    )
    assert "photonic crosstalk parity" in output
    assert "15" in output


def test_mojo_pair_and_batch_abi_match_python_atomically(tmp_path: Path) -> None:
    """Build the Mojo C ABI and compare valid and rejected calls to Python."""
    library_path = tmp_path / "libphotonic_emitter.so"
    _run(
        pin_isa(
            [
                _require_tool("mojo"),
                "build",
                "--emit",
                "shared-lib",
                "-o",
                str(library_path),
                str(_MOJO_SOURCE),
            ]
        )
    )
    library = ctypes.CDLL(str(library_path))
    pair_kernel = library.photonic_crosstalk_pair_c
    pair_kernel.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
    ]
    pair_kernel.restype = ctypes.c_longlong

    output = (ctypes.c_double * 3)(-7.0, -7.0, -7.0)
    status = pair_kernel(200.0, 50.0, 1550.0, 3.48, 1.45, ctypes.addressof(output))
    reference = WaveguidePair(gap_nm=200.0, coupling_length_um=50.0)
    assert status == 0
    expected_pair = [
        reference.coupling_coefficient,
        reference.coupling_ratio,
        reference.isolation_db,
    ]
    np.testing.assert_allclose(list(output)[:2], expected_pair[:2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(list(output)[2:], expected_pair[2:], rtol=1.0e-11, atol=1.0e-11)

    before = tuple(output)
    assert pair_kernel(float("nan"), 50.0, 1550.0, 3.48, 1.45, ctypes.addressof(output)) == -1
    assert tuple(output) == before

    batch_kernel = library.photonic_crosstalk_batch_c
    batch_kernel.argtypes = [
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
    ]
    batch_kernel.restype = ctypes.c_longlong
    gaps = (ctypes.c_double * 3)(180.0, 260.0, 520.0)
    lengths = (ctypes.c_double * 3)(8.0, 12.0, 16.0)
    batch_output = (ctypes.c_double * 9)(*([-9.0] * 9))
    status = batch_kernel(
        ctypes.addressof(gaps),
        ctypes.addressof(lengths),
        3,
        1550.0,
        3.48,
        1.45,
        ctypes.addressof(batch_output),
    )
    assert status == 0
    expected: list[float] = []
    for gap, length in zip(gaps, lengths):
        pair = WaveguidePair(gap_nm=gap, coupling_length_um=length)
        expected.extend([pair.coupling_coefficient, pair.coupling_ratio, pair.isolation_db])
    actual_matrix = np.asarray(batch_output).reshape(-1, 3)
    expected_matrix = np.asarray(expected).reshape(-1, 3)
    np.testing.assert_allclose(actual_matrix[:, :2], expected_matrix[:, :2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        actual_matrix[:, 2], expected_matrix[:, 2], rtol=1.0e-11, atol=1.0e-11
    )

    lengths[1] = -1.0
    before_batch = tuple(batch_output)
    assert (
        batch_kernel(
            ctypes.addressof(gaps),
            ctypes.addressof(lengths),
            3,
            1550.0,
            3.48,
            1.45,
            ctypes.addressof(batch_output),
        )
        == -1
    )
    assert tuple(batch_output) == before_batch
