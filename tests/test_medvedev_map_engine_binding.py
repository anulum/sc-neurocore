# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev map engine-binding contracts

"""Installed-extension contracts for the Medvedev map binding."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import medvedev_map
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = MedvedevMapNeuron()
    return (
        neuron.u,
        neuron.beta_0,
        neuron.beta_hc,
        neuron.beta_sn,
        neuron.delta,
        neuron.decay_t0,
        neuron.alpha_t0,
        neuron.f_0,
        neuron.f_1,
        neuron.homoclinic_exponent,
        neuron.d,
        neuron.input_gain,
    )


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float]:
    result: tuple[Any, int, float] = extension.py_medvedev_map_simulate(
        *_parameters(), n_steps, 0.0
    )
    trace, events, final_state = result
    return np.asarray(trace, dtype=np.float64), int(events), float(final_state)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_medvedev_map_simulate

    assert function.__name__ == "py_medvedev_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(u0, beta_0, beta_hc, beta_sn, delta, decay_t0, alpha_t0, f_0, f_1, "
        "homoclinic_exponent, d, input_gain, n_steps, current)"
    )
    assert engine.py_medvedev_map_simulate is function
    assert "py_medved_map_simulate" not in engine.__all__
    assert "py_medvedev_map_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_events, empty_state = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_events, empty_state) == (0, 0.2514078836724436)

    one_trace, one_events, one_state = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([0.1820152787145665], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_events, one_state) == (1, 0.1820152787145665)

    three_trace, three_events, three_state = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [0.1820152787145665, 0.19448491761002404, 0.206681849037328],
            dtype=np.float64,
        ),
    )
    assert (three_events, three_state) == (3, 0.206681849037328)


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
        extension.py_medvedev_map_simulate(*_parameters(), n_steps, 0.0)
    assert str(captured.value) == message
    assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert medvedev_map._HAS_RUST is True
    assert medvedev_map._rust_simulate is engine.py_medvedev_map_simulate

    rust_neuron = MedvedevMapNeuron()
    python_neuron = MedvedevMapNeuron()
    rust_trace, rust_events = rust_neuron.simulate(500, 2.0, backend="rust")
    python_trace, python_events = python_neuron.simulate(500, 2.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_events == python_events
    assert rust_neuron.u == python_neuron.u
