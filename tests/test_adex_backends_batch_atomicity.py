# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx complete-batch atomicity contracts

"""Real C-ABI and public-packet failure atomicity for AdEx."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import sc_neurocore.neurons.models.adex as adex
from sc_neurocore.neurons.models.adex import AdExNeuron
from tests.adex_backends_support import _c_arguments, _require_adex_backend


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_complete_c_abi_leaves_all_buffers_untouched_on_late_failure(backend: str) -> None:
    """Reject a second-step adaptation overflow before exposing any first row."""
    _require_adex_backend(backend)
    neuron = AdExNeuron(b=1.0e308)
    v_trace = np.full(3, -991.0, dtype=np.float64)
    w_trace = np.full(3, -992.0, dtype=np.float64)
    events = np.full(2, 193, dtype=np.uint8)
    if backend == "go":
        assert adex._go_lib is not None
        status = adex._go_lib.adex_simulate_complete_c(
            *_c_arguments(neuron),
            2,
            1.79e308,
            v_trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            w_trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    else:
        assert adex._mojo_lib is not None
        status = adex._mojo_lib.adex_simulate_complete_c(
            *_c_arguments(neuron),
            2,
            1.79e308,
            int(v_trace.ctypes.data),
            int(w_trace.ctypes.data),
            int(events.ctypes.data),
        )
    assert status == -1
    np.testing.assert_array_equal(v_trace, np.full(3, -991.0, dtype=np.float64))
    np.testing.assert_array_equal(w_trace, np.full(3, -992.0, dtype=np.float64))
    np.testing.assert_array_equal(events, np.full(2, 193, dtype=np.uint8))


@pytest.mark.parametrize(
    "packet",
    (
        (np.zeros(2), np.zeros(1), np.zeros(2), (0.0, 0.0)),
        (np.zeros(2), np.zeros(2), np.array([0, 2]), (0.0, 0.0)),
        (np.zeros(2), np.zeros(2), np.zeros(2), (1.0, 0.0)),
    ),
)
def test_public_packet_validation_rejects_malformed_output_without_mutation(
    packet: tuple[object, object, object, tuple[float, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never commit state from malformed native output packets."""
    neuron = AdExNeuron()
    before = (neuron.v, neuron.w)
    monkeypatch.setattr(neuron, "_simulate_python_complete", lambda _steps, _current: packet)
    with pytest.raises(FloatingPointError, match="AdEx backend"):
        neuron.simulate_complete(2, 0.0, backend="python")
    assert (neuron.v, neuron.w) == before
