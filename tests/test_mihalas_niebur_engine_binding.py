# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur engine-binding contracts

"""Installed-extension contracts for the Mihalas-Niebur batch binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import mihalas_niebur
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = MihalasNieburNeuron()
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


def _direct(
    n_steps: int, current: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, float, float, float, float]:
    trace, spikes, final_v, final_theta, final_i1, final_i2 = extension.py_mihalas_niebur_simulate(
        *_parameters(), n_steps, current
    )
    return (
        np.asarray(trace),
        int(spikes),
        float(final_v),
        float(final_theta),
        float(final_i1),
        float(final_i2),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_mihalas_niebur_simulate

    assert function.__name__ == "py_mihalas_niebur_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, theta0, i1_0, i2_0, v_rest, v_reset, theta_reset, theta_inf, tau_v, "
        "tau_theta, tau_1, tau_2, a, b, r1, r2, dt, n_steps, current)"
    )
    assert engine.py_mihalas_niebur_simulate is function
    assert "py_mihalas_niebur_simulate" in engine.__all__


def test_empty_and_tonic_trajectory_contracts_are_exact() -> None:
    empty = _direct(0, 2.0)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[0].flags.c_contiguous
    assert empty[1:] == (0, 0.0, 1.0, 0.0, 0.0)

    actual = _direct(8, 3.0)
    np.testing.assert_array_equal(
        actual[0],
        [
            0.2854875,
            0.5438072957812501,
            0.7775447339964668,
            0.989039133247528,
            0.0,
            0.2854875,
            0.5438072957812501,
            0.7775447339964668,
        ],
    )
    assert actual[1:] == (1, 0.7775447339964668, 1.0, 0.0, 0.0)


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
        extension.py_mihalas_niebur_simulate(*_parameters(), n_steps, 2.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_dispatcher_is_installed_and_bit_exact() -> None:
    assert mihalas_niebur._HAS_RUST is True
    assert mihalas_niebur._rust_simulate is engine.py_mihalas_niebur_simulate

    rust_neuron = MihalasNieburNeuron()
    python_neuron = MihalasNieburNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(4096, 2.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(4096, 2.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (
        rust_neuron.v,
        rust_neuron.theta,
        rust_neuron.i1,
        rust_neuron.i2,
    ) == (
        python_neuron.v,
        python_neuron.theta,
        python_neuron.i1,
        python_neuron.i2,
    )
