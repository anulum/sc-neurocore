# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real EscapeRate backend loading contracts

"""Build and resolve every maintained seeded EscapeRate execution surface."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess

import numpy as np

from sc_neurocore.accel import escape_rate as backends

_REPOSITORY = Path(__file__).resolve().parents[1]


def _contract() -> tuple[float | int, ...]:
    return (-70.0, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0, 1.0, 0xACE1, 1, 0.0)


def test_rust_engine_package_reexports_the_seeded_batch() -> None:
    """Reach the production PyO3 batch through the installed package boundary."""
    import sc_neurocore_engine

    assert backends._HAS_RUST
    assert callable(sc_neurocore_engine.py_escape_rate_simulate)
    trace, events, final_v, final_rng = sc_neurocore_engine.py_escape_rate_simulate(*_contract())
    assert np.asarray(trace).shape == (1,)
    assert np.asarray(events).tolist() == [0]
    assert final_v == -70.0
    assert 1 <= final_rng <= 0xFFFF


def test_rust_stateful_reset_replays_the_rng_state() -> None:
    """The same-name PyO3 class exposes and restores its explicit seed."""
    import sc_neurocore_engine

    neuron = sc_neurocore_engine.EscapeRateNeuron(seed=42)
    for _ in range(100):
        neuron.step(30.0)
    neuron.reset()
    state = neuron.get_state()
    assert state["v"] == -70.0
    assert state["rng_state"] == state["initial_seed"] == 42


def test_julia_loader_executes_the_committed_module() -> None:
    """Load Julia source and execute the complete state/RNG batch."""
    assert backends.ensure_julia_loaded()
    trace, events, final_v, final_rng = backends.simulate_julia(*_contract())
    assert trace.tolist() == [-70.0]
    assert events.tolist() == [0]
    assert final_v == -70.0
    assert 1 <= final_rng <= 0xFFFF


def test_go_c_shared_build_reproduces_the_committed_header(tmp_path: Path) -> None:
    """Build the cgo package and keep its public C ABI reproducible."""
    source = _REPOSITORY / "src/sc_neurocore/accel/go/neurons/escape_rate"
    output = tmp_path / "libescape_rate.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.escape_rate_simulate_c is not None
    assert output.with_suffix(".h").read_bytes() == (source / "libescape_rate.h").read_bytes()


def test_mojo_shared_build_exports_the_real_batch_symbol(tmp_path: Path) -> None:
    """Compile the maintained Mojo source and resolve its exported C ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/escape_rate.mojo"
    output = tmp_path / "libescape_rate.so"
    subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", "-o", str(output), str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.escape_rate_simulate_c is not None


def test_staged_go_and_mojo_artifacts_reload_through_dispatcher() -> None:
    """Resolve actual staged libraries through the production loaders."""
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()
    assert backends._go_lib is not None
    assert backends._mojo_lib is not None
