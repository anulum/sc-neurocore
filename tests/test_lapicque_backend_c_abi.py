# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque C-ABI rejection contracts

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import lapicque as backends
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from tests.lapicque_backends_support import c_arguments


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("current", (math.nan, 1.0e308))
def test_c_abi_rejects_invalid_run_without_writing_output(
    backend: str,
    current: float,
) -> None:
    """Reject invalid input or candidates before emitting any caller-visible row."""
    neuron = LapicqueNeuron(
        v=0.25,
        v_threshold=1.0e308,
        resistance=1.0e308 if math.isfinite(current) else 1.0,
    )
    output = np.full(2, -999.0, dtype=np.float64)
    if backend == "go":
        if not backends.ensure_go_loaded():
            pytest.skip("Go Lapicque backend is not built in this environment")
        assert backends._go_lib is not None
        result = backends._go_lib.lapicque_simulate_c(
            *c_arguments(neuron),
            1,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        if not backends.ensure_mojo_loaded():
            pytest.skip("Mojo Lapicque backend is not built in this environment")
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.lapicque_simulate_c(
            *c_arguments(neuron), 1, current, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into a mutation-free failure."""
    neuron = LapicqueNeuron(v=0.25, v_threshold=1.0e308, resistance=1.0e308)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert neuron.v == 0.25


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_complete_c_abi_rejects_source_batch_without_writing_either_buffer(
    backend: str,
) -> None:
    """Prove two-pass atomicity for the state and event custody buffers."""
    loader = backends.ensure_go_loaded if backend == "go" else backends.ensure_mojo_loaded
    if not loader():
        pytest.skip(f"{backend} Lapicque backend is not built in this environment")
    library = backends._go_lib if backend == "go" else backends._mojo_lib
    assert library is not None
    voltage = np.full(3, -777.0, dtype=np.float64)
    events = np.full(2, 255, dtype=np.uint8)
    tail: tuple[Any, Any] = (
        voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if backend == "mojo":
        tail = (int(voltage.ctypes.data), int(events.ctypes.data))
    result = library.lapicque_simulate_complete_c(
        0.0,
        0.0,
        0.0,
        1.0,
        20.0,
        1.0,
        0.01,
        1.1,
        10.0,
        1.0,
        0,
        1,
        2,
        math.nan,
        *tail,
    )
    assert result == -1
    np.testing.assert_array_equal(voltage, np.full(3, -777.0, dtype=np.float64))
    np.testing.assert_array_equal(events, np.full(2, 255, dtype=np.uint8))
