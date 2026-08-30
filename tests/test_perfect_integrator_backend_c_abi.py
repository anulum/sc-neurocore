# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator C ABI rejection contracts

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from sc_neurocore.accel import perfect_integrator as backends
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from tests.perfect_integrator_backends_support import c_arguments


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("current", (math.nan, 1.0e308))
def test_c_abi_rejects_invalid_run_without_writing_output(
    backend: str,
    current: float,
) -> None:
    """Reject invalid work before emitting any caller-visible row."""
    neuron = PerfectIntegratorNeuron(
        v=0.25,
        v_threshold=1.0e308,
        c_m=1.0e-308 if math.isfinite(current) else 1.0,
    )
    output = np.full(2, -999.0, dtype=np.float64)
    events = np.full(1, 255, dtype=np.uint8)
    if backend == "go":
        if not backends.ensure_go_loaded():
            pytest.skip("Go Perfect Integrator backend is not built in this environment")
        assert backends._go_lib is not None
        result = backends._go_lib.perfect_integrator_simulate_complete_c(
            *c_arguments(neuron),
            1,
            1,
            current,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    else:
        if not backends.ensure_mojo_loaded():
            pytest.skip("Mojo Perfect Integrator backend is not built in this environment")
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.perfect_integrator_simulate_complete_c(
            *c_arguments(neuron),
            1,
            1,
            current,
            int(output.ctypes.data),
            int(events.ctypes.data),
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))
    np.testing.assert_array_equal(events, np.full(1, 255, dtype=np.uint8))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into mutation-free failure."""
    neuron = PerfectIntegratorNeuron(v=0.25, v_threshold=1.0e308, c_m=1.0e-308)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, 1.0e308, backend=backend)
    assert neuron.v == 0.25
