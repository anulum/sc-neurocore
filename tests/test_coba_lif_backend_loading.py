# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real COBA LIF backend loading contracts

"""Build and load the production Rust, Julia, Go, and Mojo execution surfaces."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.accel import coba_lif as backends
from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_REPOSITORY = Path(__file__).resolve().parents[1]


def test_rust_engine_package_reexports_the_configurable_batch() -> None:
    """Reach the production PyO3 batch through the installed package boundary."""
    import sc_neurocore_engine

    assert backends._HAS_RUST
    assert callable(sc_neurocore_engine.py_coba_lif_simulate)
    result = sc_neurocore_engine.py_coba_lif_simulate(
        -60.0,
        0.0,
        0.0,
        0.0,
        200.0,
        10.0,
        -60.0,
        0.0,
        -80.0,
        5.0,
        10.0,
        -50.0,
        -60.0,
        5.0,
        0.1,
        1,
        0.0,
        0.0,
        0.0,
    )
    assert result[1:] == (0, -60.0, 0.0, 0.0, 0.0)


def test_rust_stateful_constructor_rejects_invalid_contract() -> None:
    """Match the Python constructor's eager validation at the PyO3 boundary."""
    import sc_neurocore_engine

    with pytest.raises(ValueError, match="invalid COBA LIF state"):
        sc_neurocore_engine.COBALIFNeuron(g_e=-1.0)


def test_julia_loader_executes_the_committed_module() -> None:
    """Load Julia source and execute its full-state batch without a surrogate."""
    assert backends.ensure_julia_loaded()
    trace, spikes, state = backends.simulate_julia(
        -60.0,
        0.0,
        0.0,
        0.0,
        200.0,
        10.0,
        -60.0,
        0.0,
        -80.0,
        5.0,
        10.0,
        -50.0,
        -60.0,
        5.0,
        0.1,
        1,
        0.0,
        0.0,
        0.0,
    )
    assert trace.tolist() == [-60.0]
    assert spikes == 0
    assert state == (-60.0, 0.0, 0.0, 0.0)


def test_go_c_shared_build_reproduces_the_committed_header(tmp_path: Path) -> None:
    """Build the real cgo package and keep its public C ABI reproducible."""
    source = _REPOSITORY / "src/sc_neurocore/accel/go/neurons/coba_lif"
    output = tmp_path / "libcoba_lif.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    library = ctypes.CDLL(str(output))
    assert library.coba_lif_simulate_c is not None
    assert output.with_suffix(".h").read_bytes() == (source / "libcoba_lif.h").read_bytes()


def test_mojo_shared_build_exports_the_real_batch_symbol(tmp_path: Path) -> None:
    """Compile the maintained Mojo source and resolve its exported C ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/coba_lif.mojo"
    output = tmp_path / "libcoba_lif.so"
    subprocess.run(
        pin_isa(["mojo", "build", "--emit", "shared-lib", "-o", str(output), str(source)]),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    library = ctypes.CDLL(str(output))
    assert library.coba_lif_simulate_c is not None


def test_staged_go_and_mojo_artifacts_reload_through_dispatcher() -> None:
    """Resolve the actual staged libraries through production loader functions."""
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()
    assert backends._go_lib is not None
    assert backends._mojo_lib is not None
