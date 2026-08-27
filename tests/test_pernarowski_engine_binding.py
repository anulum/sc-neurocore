# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski engine-binding contracts

"""Installed-extension contracts for the Pernarowski binding."""

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
from sc_neurocore.neurons.models import pernarowski
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = PernarowskiNeuron()
    return (
        neuron.v,
        neuron.w,
        neuron.z,
        neuron.alpha,
        neuron.beta,
        neuron.eps1,
        neuron.eps2,
        neuron.gamma,
        neuron.dt,
        neuron.v_threshold,
    )


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float, float]:
    result: tuple[Any, int, float, float, float] = extension.py_pernarowski_simulate(
        *_parameters(), n_steps, 0.0
    )
    trace, spikes, final_v, final_w, final_z = result
    return (
        np.asarray(trace, dtype=np.float64),
        int(spikes),
        float(final_v),
        float(final_w),
        float(final_z),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_pernarowski_simulate

    assert function.__name__ == "py_pernarowski_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, w0, z0, alpha, beta, eps1, eps2, gamma, dt, v_threshold, n_steps, current)"
    )
    assert engine.py_pernarowski_simulate is function
    assert "py_pernarowski_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_v, empty_w, empty_z = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_v, empty_w, empty_z) == (0, -1.0, 0.0, 0.0)

    one_trace, one_spikes, one_v, one_w, one_z = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-1.06605689484945], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_v, one_w, one_z) == (
        0,
        -1.06605689484945,
        -0.009308418411175487,
        -1.6656362070103914e-05,
    )

    three_trace, three_spikes, three_v, three_w, three_z = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array([-1.06605689484945, -1.1302625842924157, -1.1917060311305951]),
    )
    assert (three_spikes, three_v, three_w, three_z) == (
        0,
        -1.1917060311305951,
        -0.029711278988164558,
        -5.9629107101179414e-05,
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
        extension.py_pernarowski_simulate(*_parameters(), n_steps, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert pernarowski._HAS_RUST is True
    assert pernarowski._rust_simulate is engine.py_pernarowski_simulate

    rust_neuron = PernarowskiNeuron()
    python_neuron = PernarowskiNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 0.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 0.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.v, rust_neuron.w, rust_neuron.z) == (
        python_neuron.v,
        python_neuron.w,
        python_neuron.z,
    )
