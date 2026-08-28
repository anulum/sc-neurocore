# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained resetting Wilson-HR engine binding

"""Installed PyO3 and NetworkRunner contracts for the retained SC identity."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models import sc_resetting_wilson_hr as implementation
from sc_neurocore.neurons.models.sc_resetting_wilson_hr import SCResettingWilsonHRNeuron
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_native_batch_export_and_signature_are_stable() -> None:
    function = extension.py_sc_resetting_wilson_hr_simulate
    assert function.__name__ == "py_sc_resetting_wilson_hr_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == "(v0, r0, tau_r, v_peak, dt, n_steps, current)"
    assert engine.py_sc_resetting_wilson_hr_simulate is function
    assert "py_sc_resetting_wilson_hr_simulate" in engine.__all__


def test_native_initial_updates_preserve_historical_anchor() -> None:
    result: tuple[Any, int, float, float] = extension.py_sc_resetting_wilson_hr_simulate(
        -0.7, 0.1, 1.9, 0.4, 0.05, 1, 2.0
    )
    trace, events, final_v, final_r = result
    np.testing.assert_array_equal(
        np.asarray(trace, dtype=np.float64),
        np.array([-0.5988676025214146], dtype=np.float64),
    )
    assert (events, final_v, final_r) == (
        0,
        -0.5988676025214146,
        0.10134793845659071,
    )


def test_production_rust_batch_matches_public_python_model() -> None:
    assert implementation._HAS_RUST is True
    rust_model = SCResettingWilsonHRNeuron()
    python_model = SCResettingWilsonHRNeuron()
    rust_trace, rust_events = rust_model.simulate(1_000, 2.0, backend="rust")
    python_trace, python_events = python_model.simulate(1_000, 2.0, backend="python")
    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_events == python_events
    assert (rust_model.v, rust_model.r) == (python_model.v, python_model.r)


def test_native_invalid_batch_fails_explicitly() -> None:
    with pytest.raises(FloatingPointError, match="rejected an invalid candidate"):
        extension.py_sc_resetting_wilson_hr_simulate(1.0e103, 0.1, 1.9, 0.4, 0.05, 2, 2.0)


def test_network_runner_executes_distinct_sc_model() -> None:
    assert "SCResettingWilsonHR" in extension.NetworkRunner.supported_models()
    runner = extension.NetworkRunner()
    population = runner.add_population("SCResettingWilsonHR", 1)
    result = runner.step_population(population, np.array([2.0], dtype=np.float64))
    assert result["spikes"].tolist() == [0]
    assert result["voltages"].tolist() == [-0.5988676025214146]
