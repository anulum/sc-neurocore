# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBA-LIF engine-binding contracts

"""Installed-extension contracts for the conductance-based LIF binding."""

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
from sc_neurocore.accel import coba_lif as backends
from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = COBALIFNeuron(
        v=-59.0,
        g_e=1.25,
        g_i=0.75,
        c_m=190.0,
        g_l=11.0,
        e_l=-61.0,
        e_e=2.0,
        e_i=-82.0,
        tau_e=4.5,
        tau_i=11.0,
        v_threshold=-50.5,
        v_reset=-62.0,
        refractory_period=3.7,
    )
    return (
        neuron.v,
        neuron.g_e,
        neuron.g_i,
        neuron.refractory_time,
        neuron.c_m,
        neuron.g_l,
        neuron.e_l,
        neuron.e_e,
        neuron.e_i,
        neuron.tau_e,
        neuron.tau_i,
        neuron.v_threshold,
        neuron.v_reset,
        neuron.refractory_period,
        neuron.dt,
    )


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float, float, float]:
    result: tuple[Any, int, float, float, float, float] = extension.py_coba_lif_simulate(
        *_parameters(), n_steps, 650.0, 0.15, 0.07
    )
    trace, spikes, final_v, final_ge, final_gi, final_refractory = result
    return (
        np.asarray(trace, dtype=np.float64),
        int(spikes),
        float(final_v),
        float(final_ge),
        float(final_gi),
        float(final_refractory),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_coba_lif_simulate

    assert function.__name__ == "py_coba_lif_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, g_e0, g_i0, refractory_time0, c_m, g_l, e_l, e_e, e_i, tau_e, "
        "tau_i, v_threshold, v_reset, refractory_period, dt, n_steps, current, "
        "delta_ge, delta_gi)"
    )
    assert engine.py_coba_lif_simulate is function
    assert "py_coba_lif_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, *empty_state = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, *empty_state) == (0, *_parameters()[:4])

    one_trace, one_spikes, *one_state = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-58.6361686725916], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, *one_state) == (
        0,
        -58.6361686725916,
        1.3692320215414315,
        0.8125792363966488,
        0.0,
    )

    three_trace, three_spikes, *three_state = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [-58.6361686725916, -58.271886913941316, -57.907301339762405],
            dtype=np.float64,
        ),
    )
    assert (three_spikes, *three_state) == (
        0,
        -57.907301339762405,
        1.5998925209831174,
        0.9360438621137919,
        0.0,
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
        extension.py_coba_lif_simulate(*_parameters(), n_steps, 0.0, 0.0, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            (-60.0, -1.0, 0.0, 0.0, *_parameters()[4:], 1, 0.0, 0.0, 0.0),
            "invalid COBA LIF state or parameter contract",
        ),
        ((*_parameters(), 1, np.nan, 0.0, 0.0), "invalid COBA LIF simulation input"),
        ((*_parameters(), 1, 0.0, -1.0, 0.0), "invalid COBA LIF simulation input"),
        (
            (-60.0, 1.0e9, 0.0, 0.0, *_parameters()[4:], 1, 0.0, 1.0, 0.0),
            "conductance candidate outside COBA LIF safety envelope",
        ),
        (
            (90.0, 0.0, 0.0, 0.0, *_parameters()[4:], 1, 1.0e8, 0.0, 0.0),
            "voltage candidate outside COBA LIF safety envelope",
        ),
    ),
)
def test_invalid_contracts_keep_the_exact_public_error(
    arguments: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        extension.py_coba_lif_simulate(*arguments)


def test_production_rust_backend_is_the_installed_extension_with_python_parity() -> None:
    assert backends._HAS_RUST is True
    assert backends._engine_coba_simulate is engine.py_coba_lif_simulate

    for current, delta_ge, delta_gi in ((0.0, 0.0, 0.0), (650.0, 0.15, 0.07)):
        rust_neuron = COBALIFNeuron(v=-59.0, g_e=1.25, g_i=0.75)
        python_neuron = COBALIFNeuron(v=-59.0, g_e=1.25, g_i=0.75)
        rust_trace, rust_spikes = rust_neuron.simulate(
            400, current, delta_ge, delta_gi, backend="rust"
        )
        python_trace, python_spikes = python_neuron.simulate(
            400, current, delta_ge, delta_gi, backend="python"
        )

        np.testing.assert_allclose(rust_trace, python_trace, rtol=0.0, atol=1.0e-13)
        assert rust_spikes == python_spikes
        np.testing.assert_allclose(
            (
                rust_neuron.v,
                rust_neuron.g_e,
                rust_neuron.g_i,
                rust_neuron.refractory_time,
            ),
            (
                python_neuron.v,
                python_neuron.g_e,
                python_neuron.g_i,
                python_neuron.refractory_time,
            ),
            rtol=0.0,
            atol=1.0e-13,
        )
