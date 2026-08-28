# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 engine-binding contracts

"""Installed-extension contracts for the Izhikevich 2007 batch binding."""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import izhikevich2007
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = Izhikevich2007Neuron()
    return (
        neuron.v,
        neuron.u,
        neuron.C,
        neuron.k,
        neuron.vr,
        neuron.vt,
        neuron.vpeak,
        neuron.a,
        neuron.b,
        neuron.c,
        neuron.d,
        neuron.dt,
    )


def _direct(
    n_steps: int, current: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, float, float]:
    trace, spikes, final_v, final_u = extension.py_izhikevich2007_simulate(
        *_parameters(), n_steps, current
    )
    return np.asarray(trace), int(spikes), float(final_v), float(final_u)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_izhikevich2007_simulate

    assert function.__name__ == "py_izhikevich2007_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, u0, cap, k, vr, vt, vpeak, a, b, c, d, dt, n_steps, current)"
    )
    assert engine.py_izhikevich2007_simulate is function
    assert "py_izhikevich2007_simulate" in engine.__all__


def test_empty_and_early_trajectory_contracts_are_exact() -> None:
    empty = _direct(0, 300.0)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[0].flags.c_contiguous
    assert empty[1:] == (0, -60.0, -0.0)

    actual = _direct(8, 300.0)
    np.testing.assert_array_equal(
        actual[0],
        [
            -59.70206922640671,
            -59.408156359519694,
            -59.11808610199188,
            -58.83169016224595,
            -58.54880683232725,
            -58.26928059281069,
            -57.9929617426314,
            -57.71970605189161,
        ],
    )
    assert actual[1:] == (0, -57.71970605189161, -0.05518261740479853)


@pytest.mark.parametrize(
    ("n_steps", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_step_count_conversion_errors_are_stable(
    n_steps: object, error: type[BaseException], message: str
) -> None:
    with pytest.raises(error) as captured:
        extension.py_izhikevich2007_simulate(*_parameters(), n_steps, 300.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_spike_reset_trace_and_final_state_are_exact() -> None:
    trace, spikes, final_v, final_u = _direct(500, 300.0)

    assert spikes == 3
    np.testing.assert_array_equal(np.flatnonzero(trace == -50.0), [145, 276, 433])
    assert (final_v, final_u) == (-43.89897660053505, 128.4117914934368)


def test_production_rust_dispatcher_is_installed_and_bit_exact() -> None:
    assert izhikevich2007._HAS_RUST is True
    assert izhikevich2007._rust_simulate is engine.py_izhikevich2007_simulate

    rust_neuron = Izhikevich2007Neuron()
    python_neuron = Izhikevich2007Neuron()
    rust_trace, rust_spikes = rust_neuron.simulate(8_000, 300.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(8_000, 300.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.v, rust_neuron.u) == (python_neuron.v, python_neuron.u)


def test_direct_binding_rejects_invalid_input() -> None:
    with pytest.raises(FloatingPointError, match="invalid Izhikevich 2007"):
        extension.py_izhikevich2007_simulate(*_parameters(), 1, math.nan)


def test_network_runner_exposes_the_distinct_2007_identity() -> None:
    assert "Izhikevich" in extension.NetworkRunner.supported_models()
    assert "Izhikevich2007" in extension.NetworkRunner.supported_models()
    runner = extension.NetworkRunner()
    population = runner.add_population("Izhikevich2007", 1)
    result = runner.step_population(population, np.array([300.0], dtype=np.float64))
    assert result["spikes"].tolist() == [0]
    assert result["voltages"].tolist() == [-59.70206922640671]
