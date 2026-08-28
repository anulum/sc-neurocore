# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF5 installed-extension contracts

"""Real PyO3 export and dispatcher contracts for canonical GLIF5."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import glif
from sc_neurocore.neurons.models.glif import GLIFNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _arguments() -> tuple[float, ...]:
    neuron = GLIFNeuron()
    return (
        neuron.v,
        neuron.theta_spike,
        neuron.i_asc1,
        neuron.i_asc2,
        neuron.theta_voltage,
        neuron.refractory_remaining,
        neuron.e_l,
        neuron.capacitance,
        neuron.resistance,
        neuron.theta_inf,
        neuron.b_spike,
        neuron.b_voltage,
        neuron.a_voltage,
        neuron.k_asc1,
        neuron.k_asc2,
        neuron.f_v,
        neuron.delta_v,
        neuron.delta_theta_spike,
        neuron.f_asc1,
        neuron.f_asc2,
        neuron.delta_i_asc1,
        neuron.delta_i_asc2,
        neuron.refractory_period,
        neuron.dt,
    )


def _direct(n_steps: int, current: float) -> tuple[Any, ...]:
    result = extension.py_glif_simulate(*_arguments(), n_steps, current)
    return (np.asarray(result[0]), int(result[1]), *(float(value) for value in result[2:]))


def test_exported_name_and_top_level_identity_are_stable() -> None:
    function = extension.py_glif_simulate

    assert function.__name__ == "py_glif_simulate"
    assert engine.py_glif_simulate is function
    assert "py_glif_simulate" in engine.__all__
    assert "theta_spike0" in function.__text_signature__
    assert "refractory_period" in function.__text_signature__


def test_empty_and_driven_batches_return_complete_state() -> None:
    empty = _direct(0, 30.0)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[1:] == (0, -70.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    driven = _direct(1000, 30.0)
    assert driven[0].shape == (1000,)
    assert driven[1] == 49
    assert driven[-1] == 1.0


def test_direct_binding_rejects_invalid_numeric_contract() -> None:
    arguments = list(_arguments())
    arguments[7] = 0.0
    with pytest.raises(FloatingPointError, match="invalid candidate"):
        extension.py_glif_simulate(*arguments, 1, 30.0)


def test_production_dispatcher_is_installed_and_exact() -> None:
    assert glif._HAS_RUST is True
    assert glif._rust_simulate is engine.py_glif_simulate
    expected = GLIFNeuron()
    actual = GLIFNeuron()
    expected_trace, expected_events = expected.simulate(4096, 30.0, backend="python")
    actual_trace, actual_events = actual.simulate(4096, 30.0, backend="rust")

    np.testing.assert_array_equal(actual_trace, expected_trace)
    assert actual_events == expected_events
    assert vars(actual) == vars(expected)
