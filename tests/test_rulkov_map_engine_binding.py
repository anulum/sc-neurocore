# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map engine-binding contracts

"""Installed-extension contracts for the Rulkov map binding."""

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
from sc_neurocore.neurons.models import rulkov_map
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_rulkov_map_simulate(
        -1.0,
        -3.0,
        4.0,
        -1.6,
        0.001,
        n_steps,
        0.0,
    )
    trace, spikes, x_final, y_final = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(x_final), float(y_final)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_rulkov_map_simulate

    assert function.__name__ == "py_rulkov_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == "(x0, y0, alpha, sigma, mu, n_steps, current)"
    assert engine.py_rulkov_map_simulate is function
    assert "py_rulkov_map_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_x, empty_y = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_x, empty_y) == (0, -1.0, -3.0)

    one_trace, one_spikes, one_x, one_y = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-1.0], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_x, one_y) == (0, -1.0, -3.0016)

    three_trace, three_spikes, three_x, three_y = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array([-1.0, -1.0015999999999998, -1.004798721023181], dtype=np.float64),
    )
    assert (three_spikes, three_x, three_y) == (
        0,
        -1.004798721023181,
        -3.0047983999999994,
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
        extension.py_rulkov_map_simulate(
            -1.0,
            -3.0,
            4.0,
            -1.6,
            0.001,
            n_steps,
            0.0,
        )
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert rulkov_map._HAS_RUST is True
    assert rulkov_map._rust_simulate is engine.py_rulkov_map_simulate

    rust_neuron = RulkovMapNeuron()
    python_neuron = RulkovMapNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 0.5, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 0.5, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.x, rust_neuron.y) == (python_neuron.x, python_neuron.y)
