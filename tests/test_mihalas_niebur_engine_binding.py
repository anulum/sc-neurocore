# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur production engine binding

"""Installed PyO3 contracts for the source Mihalas-Niebur batch kernel."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import mihalas_niebur
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _arguments(neuron: MihalasNieburNeuron) -> tuple[float, ...]:
    return (
        neuron.v,
        neuron.theta,
        neuron.i1,
        neuron.i2,
        neuron.v_rest,
        neuron.v_reset,
        neuron.theta_reset,
        neuron.theta_inf,
        neuron.leak_rate,
        neuron.threshold_voltage_coupling,
        neuron.threshold_decay_rate,
        neuron.current_decay_rate_1,
        neuron.current_decay_rate_2,
        neuron.current_retention_1,
        neuron.current_retention_2,
        neuron.current_jump_1,
        neuron.current_jump_2,
        neuron.dt,
    )


def test_exported_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_mihalas_niebur_simulate

    assert function.__name__ == "py_mihalas_niebur_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf, "
        "leak_rate, threshold_voltage_coupling, threshold_decay_rate, "
        "current_decay_rate_1, current_decay_rate_2, current_retention_1, "
        "current_retention_2, current_jump_1, current_jump_2, dt, n_steps, current)"
    )
    assert engine.py_mihalas_niebur_simulate is function
    assert "py_mihalas_niebur_simulate" in engine.__all__


def test_direct_binding_matches_public_python_reference() -> None:
    reference = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    expected_trace, expected_events = reference.simulate(2000, 0.002, backend="python")
    bound = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    result = extension.py_mihalas_niebur_simulate(*_arguments(bound), 2000, 0.002)
    trace, events, *state = result

    np.testing.assert_array_equal(np.asarray(trace), expected_trace)
    assert int(events) == expected_events == 14
    assert tuple(float(value) for value in state) == (
        reference.v,
        reference.theta,
        reference.i1,
        reference.i2,
    )


def test_direct_binding_rejects_invalid_candidates() -> None:
    neuron = MihalasNieburNeuron()
    arguments: list[Any] = list(_arguments(neuron))

    with pytest.raises(FloatingPointError, match="invalid candidate"):
        extension.py_mihalas_niebur_simulate(*arguments, 4, float("nan"))


def test_production_dispatcher_uses_installed_binding() -> None:
    assert mihalas_niebur._HAS_RUST is True
    assert mihalas_niebur._rust_simulate is engine.py_mihalas_niebur_simulate

    rust = MihalasNieburNeuron()
    python = MihalasNieburNeuron()
    rust_trace, rust_events = rust.simulate(4096, 0.002, backend="rust")
    python_trace, python_events = python.simulate(4096, 0.002, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_events == python_events
    assert (rust.v, rust.theta, rust.i1, rust.i2) == (
        python.v,
        python.theta,
        python.i1,
        python.i2,
    )
