# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman-Wang engine-binding contracts

"""Installed-extension contracts for the Terman-Wang binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import terman_wang
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = TermanWangOscillator()
    return (
        neuron.v,
        neuron.w,
        neuron.alpha,
        neuron.beta,
        neuron.epsilon,
        neuron.rho,
        neuron.dt,
        neuron.v_peak,
    )


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_terman_wang_simulate(
        *_parameters(), n_steps, 0.0
    )
    trace, spikes, final_v, final_w = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(final_v), float(final_w)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_terman_wang_simulate

    assert function.__name__ == "py_terman_wang_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, w0, alpha, beta, epsilon, rho, dt, v_peak, n_steps, current)"
    )
    assert engine.py_terman_wang_simulate is function
    assert "py_terman_wang_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_v, empty_w = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_v, empty_w) == (0, -1.5, -0.5)

    one_trace, one_spikes, one_v, one_w = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-1.4370294307953222], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_v, one_w) == (0, -1.4370294307953222, -0.4995002473377955)

    three_trace, three_spikes, three_v, three_w = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [-1.4370294307953222, -1.3834201384196982, -1.3367266628606986],
            dtype=np.float64,
        ),
    )
    assert (three_spikes, three_v, three_w) == (
        0,
        -1.3367266628606986,
        -0.4985022330562414,
    )


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
        extension.py_terman_wang_simulate(*_parameters(), n_steps, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


@pytest.mark.parametrize(
    "parameters",
    (
        (*_parameters(), 3, np.nan),
        (-1.5, -0.5, 3.0, 0.0, 0.02, 0.0, 0.05, 1.5, 3, 0.0),
        (-1.5, -0.5, 3.0, 0.2, 0.02, 0.0, 0.0, 1.5, 3, 0.0),
    ),
)
def test_invalid_direct_numeric_inputs_preserve_the_initial_state(
    parameters: tuple[float, ...],
) -> None:
    trace, spikes, final_v, final_w = extension.py_terman_wang_simulate(*parameters)
    np.testing.assert_array_equal(trace, np.full(3, -1.5, dtype=np.float64))
    assert (spikes, final_v, final_w) == (0, -1.5, -0.5)


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert terman_wang._HAS_RUST is True
    assert terman_wang._rust_simulate is engine.py_terman_wang_simulate

    for current in (0.0, 0.5, 1.0, 1.5):
        rust_neuron = TermanWangOscillator()
        python_neuron = TermanWangOscillator()
        rust_trace, rust_spikes = rust_neuron.simulate(4_000, current, backend="rust")
        python_trace, python_spikes = python_neuron.simulate(4_000, current, backend="python")

        np.testing.assert_array_equal(rust_trace, python_trace)
        assert rust_spikes == python_spikes
        assert (rust_neuron.v, rust_neuron.w) == (python_neuron.v, python_neuron.w)
