# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained four-state GLIF engine-binding contracts

"""Installed-extension contracts for the retained four-state GLIF batch binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import sc_four_state_glif as glif
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = SCFourStateGLIFNeuron()
    return (
        neuron.v,
        neuron.theta,
        neuron.theta_inf,
        neuron.i_asc1,
        neuron.i_asc2,
        neuron.v_rest,
        neuron.v_reset,
        neuron.tau_m,
        neuron.tau_theta,
        neuron.tau_asc1,
        neuron.tau_asc2,
        neuron.a_theta,
        neuron.delta_theta,
        neuron.r_asc1,
        neuron.r_asc2,
        neuron.resistance,
        neuron.dt,
    )


def _direct(
    n_steps: int, current: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, float, float, float, float]:
    trace, spikes, final_v, final_theta, final_i_asc1, final_i_asc2 = (
        extension.py_sc_four_state_glif_simulate(*_parameters(), n_steps, current)
    )
    return (
        np.asarray(trace),
        int(spikes),
        float(final_v),
        float(final_theta),
        float(final_i_asc1),
        float(final_i_asc2),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_sc_four_state_glif_simulate

    assert function.__name__ == "py_sc_four_state_glif_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, theta0, theta_inf, i_asc1_0, i_asc2_0, v_rest, v_reset, tau_m, "
        "tau_theta, tau_asc1, tau_asc2, a_theta, delta_theta, r_asc1, r_asc2, "
        "resistance, dt, n_steps, current)"
    )
    assert engine.py_sc_four_state_glif_simulate is function
    assert "py_sc_four_state_glif_simulate" in engine.__all__


def test_empty_and_tonic_trajectory_contracts_are_exact() -> None:
    empty = _direct(0, 30.0)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[0].flags.c_contiguous
    assert empty[1:] == (0, -70.0, -50.0, 0.0, 0.0)

    actual = _direct(8, 30.0)
    np.testing.assert_array_equal(
        actual[0],
        [
            -67.14512500000001,
            -64.5619270421875,
            -62.224552660035336,
            -60.109608667524725,
            -58.1959280327014,
            -56.46435803128946,
            -54.897568560136875,
            -53.47987869203285,
        ],
    )
    assert actual[1:] == (0, -53.47987869203285, -49.99272780580647, 0.0, 0.0)


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
        extension.py_sc_four_state_glif_simulate(*_parameters(), n_steps, 30.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_dispatcher_is_installed_and_bit_exact() -> None:
    assert glif._HAS_RUST is True
    assert glif._rust_simulate is engine.py_sc_four_state_glif_simulate

    rust_neuron = SCFourStateGLIFNeuron()
    python_neuron = SCFourStateGLIFNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(4096, 30.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(4096, 30.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (
        rust_neuron.v,
        rust_neuron.theta,
        rust_neuron.i_asc1,
        rust_neuron.i_asc2,
    ) == (
        python_neuron.v,
        python_neuron.theta,
        python_neuron.i_asc1,
        python_neuron.i_asc2,
    )
