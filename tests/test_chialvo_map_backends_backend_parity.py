# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (backend_parity) from former test_chialvo_map_backends.py

"""Hold every Chialvo map backend to the checked contract, and the Go trace to the Python trace bit for bit.

The Rust, Julia and Go lanes each satisfy the complete Chialvo contract. The Go library computes its exponential with
the C library's ``exp``, so its whole trace equals the Python reference wherever NumPy's ``exp`` is that same function.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from sc_neurocore.neurons.models.chialvo_map import (
    ChialvoMapNeuron,
)
from tests.chialvo_map_backends_support import (
    _assert_backend_contract,
    _go_available,
    _julia_available,
    _mojo_available,
    _OPERATING_CURRENTS,
    _require,
    _run,
    _rust_available,
)


def test_rust_backend_contract() -> None:
    """Rust must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("rust", _rust_available)


def test_julia_backend_contract() -> None:
    """Julia must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("julia", _julia_available)


def test_go_backend_contract() -> None:
    """Go must satisfy the complete checked Chialvo contract."""
    _assert_backend_contract("go", _go_available)


def test_go_trace_is_bit_identical_where_the_reference_exp_is_the_c_library_exp() -> None:
    """The Go library uses the C library's exp, so its whole trace equals Python's wherever Python uses that exp.

    Every exponent argument of the reference orbit is checked first: NumPy's ``exp`` must equal the C library's
    (``math.exp``). On a host where they differ the claim does not apply, and the test says so by skipping.
    """
    _require("go", _go_available)
    for current in _OPERATING_CURRENTS:
        reference = _run("python", current=current)
        arguments = np.concatenate(([0.0], reference[1][:-1] - reference[0][:-1]))
        if any(float(np.exp(value)) != math.exp(value) for value in arguments):
            pytest.skip("NumPy's exp is not the C library's exp on this host")
        observed = _run("go", current=current)
        np.testing.assert_array_equal(observed[0], reference[0])
        np.testing.assert_array_equal(observed[1], reference[1])
        assert observed[2:] == reference[2:]


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
