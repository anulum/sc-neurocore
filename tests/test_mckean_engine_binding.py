# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McKean engine-binding contracts

"""Installed-extension contracts for the McKean binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import mckean
from sc_neurocore.neurons.models.mckean import McKeanNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_mckean_simulate(
        0.0, 0.0, 0.25, 0.01, 0.5, 0.1, 0.8, n_steps, 0.2
    )
    trace, spikes, final_v, final_w = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(final_v), float(final_w)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_mckean_simulate

    assert function.__name__ == "py_mckean_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, w0, a, epsilon, gamma, dt, v_peak, n_steps, current)"
    )
    assert engine.py_mckean_simulate is function
    assert "py_mckean_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_v, empty_w = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_v, empty_w) == (0, 0.0, 0.0)

    one_trace, one_spikes, one_v, one_w = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([0.019032183375], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_v, one_w) == (0, 0.019032183375, 9.673291875e-06)

    three_trace, three_spikes, three_v, three_w = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [0.019032183375, 0.03625140592518526, 0.05182855625337733],
            dtype=np.float64,
        ),
    )
    assert (three_spikes, three_v, three_w) == (
        0,
        0.05182855625337733,
        8.158904332985283e-05,
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
        extension.py_mckean_simulate(0.0, 0.0, 0.25, 0.01, 0.5, 0.1, 0.8, n_steps, 0.2)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert mckean._HAS_RUST is True
    assert mckean._rust_simulate is engine.py_mckean_simulate

    rust_neuron = McKeanNeuron()
    python_neuron = McKeanNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 0.5, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 0.5, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.v, rust_neuron.w) == (python_neuron.v, python_neuron.w)
