# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_chialvo_map_backends.py

from __future__ import annotations

from tests.chialvo_map_backends_support import *  # noqa: F403


def test_rust_backend_contract() -> None:
    """Rust must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("rust", _rust_available)


def test_julia_backend_contract() -> None:
    """Julia must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("julia", _julia_available)


def test_go_backend_contract() -> None:
    """Go must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("go", _go_available)


def test_mojo_backend_contract() -> None:
    """Mojo must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("mojo", _mojo_available)


def test_python_batch_matches_repeated_step() -> None:
    """The reference batch loop must retain the public step semantics."""
    batch = ChialvoMapNeuron()
    trace, spikes = batch.simulate(100, 0.01, backend="python")
    stepper = ChialvoMapNeuron()
    expected: npt.NDArray[np.float64] = np.empty(100, dtype=np.float64)
    expected_spikes = 0
    for index in range(100):
        expected_spikes += stepper.step(0.01)
        expected[index] = stepper.x
    np.testing.assert_array_equal(trace, expected)
    assert spikes == expected_spikes
    assert (batch.x, batch.y) == (stepper.x, stepper.y)
