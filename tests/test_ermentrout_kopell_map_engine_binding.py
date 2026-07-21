# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ermentrout-Kopell map engine-binding contracts

"""Installed-extension contracts for the Ermentrout-Kopell map binding."""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import ermentrout_kopell_map_neuron
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float]:
    result: tuple[Any, int, float] = extension.py_ermentrout_kopell_map_simulate(
        0.0, 0.1, 1.0, math.pi, n_steps, 0.1
    )
    trace, spikes, final_theta = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(final_theta)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_ermentrout_kopell_map_simulate

    assert function.__name__ == "py_ermentrout_kopell_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == ("(theta0, dt, gain, theta_threshold, n_steps, current)")
    assert engine.py_ermentrout_kopell_map_simulate is function
    assert "py_ermentrout_kopell_map_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_theta = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_theta) == (0, 0.0)

    one_trace, one_spikes, one_theta = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([0.020000000000000004]))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_theta) == (0, 0.020000000000000004)

    three_trace, three_spikes, three_theta = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [0.020000000000000004, 0.04001799940000801, 0.06009005459564935],
            dtype=np.float64,
        ),
    )
    assert (three_spikes, three_theta) == (0, 0.06009005459564935)


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
        extension.py_ermentrout_kopell_map_simulate(0.0, 0.1, 1.0, math.pi, n_steps, 0.1)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert ermentrout_kopell_map_neuron._HAS_RUST is True
    assert ermentrout_kopell_map_neuron._rust_simulate is engine.py_ermentrout_kopell_map_simulate

    rust_neuron = ErmentroutKopellMapNeuron()
    python_neuron = ErmentroutKopellMapNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 1.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 1.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert rust_neuron.theta == python_neuron.theta
