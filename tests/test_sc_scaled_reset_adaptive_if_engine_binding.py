# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained scaled-reset production engine binding

"""Installed PyO3 contracts for the count-neutral retained recurrence."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import sc_scaled_reset_adaptive_if
from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _arguments(neuron: SCScaledResetAdaptiveIFNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.theta,
        neuron.i1,
        neuron.i2,
        neuron.v_rest,
        neuron.v_reset,
        neuron.theta_reset,
        neuron.theta_inf,
        neuron.tau_v,
        neuron.tau_theta,
        neuron.tau_1,
        neuron.tau_2,
        neuron.a,
        neuron.b,
        neuron.r1,
        neuron.r2,
        neuron.dt,
    )


def test_exported_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_sc_scaled_reset_adaptive_if_simulate

    assert function.__name__ == "py_sc_scaled_reset_adaptive_if_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf, "
        "tau_v, tau_theta, tau_1, tau_2, a, b, r1, r2, dt, n_steps, current)"
    )
    assert engine.py_sc_scaled_reset_adaptive_if_simulate is function
    assert "py_sc_scaled_reset_adaptive_if_simulate" in engine.__all__


def test_direct_binding_matches_public_python_reference() -> None:
    parameters = {
        "theta_reset": 1.3,
        "tau_theta": 40.0,
        "tau_1": 15.0,
        "tau_2": 80.0,
        "a": 0.1,
        "b": 0.1,
        "r1": 0.2,
        "r2": -0.15,
    }
    reference = SCScaledResetAdaptiveIFNeuron(**parameters)
    expected_trace, expected_events = reference.simulate(1600, 3.0, backend="python")
    bound = SCScaledResetAdaptiveIFNeuron(**parameters)
    result = extension.py_sc_scaled_reset_adaptive_if_simulate(*_arguments(bound), 1600, 3.0)
    trace, events, *state = result

    np.testing.assert_array_equal(np.asarray(trace), expected_trace)
    assert int(events) == expected_events
    assert tuple(float(value) for value in state) == (
        reference.v,
        reference.theta,
        reference.i1,
        reference.i2,
    )


def test_direct_binding_rejects_invalid_candidates() -> None:
    neuron = SCScaledResetAdaptiveIFNeuron()
    arguments: list[Any] = list(_arguments(neuron))

    with pytest.raises(FloatingPointError, match="invalid candidate"):
        extension.py_sc_scaled_reset_adaptive_if_simulate(*arguments, 4, float("nan"))


def test_production_dispatcher_uses_installed_binding() -> None:
    assert sc_scaled_reset_adaptive_if._HAS_RUST is True
    assert (
        sc_scaled_reset_adaptive_if._rust_simulate is engine.py_sc_scaled_reset_adaptive_if_simulate
    )

    rust = SCScaledResetAdaptiveIFNeuron()
    python = SCScaledResetAdaptiveIFNeuron()
    rust_trace, rust_events = rust.simulate(4096, 2.0, backend="rust")
    python_trace, python_events = python.simulate(4096, 2.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_events == python_events
    assert (rust.v, rust.theta, rust.i1, rust.i2) == (
        python.v,
        python.theta,
        python.i1,
        python.i2,
    )


def test_network_runner_executes_distinct_sc_identity() -> None:
    assert "SCScaledResetAdaptiveIF" in extension.NetworkRunner.supported_models()
    runner = extension.NetworkRunner()
    population = runner.add_population("SCScaledResetAdaptiveIF", 1)
    result = runner.step_population(population, np.array([3.0], dtype=np.float64))
    assert result["spikes"].tolist() == [0]
    assert result["voltages"].tolist() == [0.2854875]
