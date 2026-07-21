# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Courbage-Nekorkin-Vdovin engine-binding contracts

"""Installed-extension contracts for the Courbage-Nekorkin-Vdovin map binding."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import courage_nekorkin_map
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_courage_nekorkin_map_simulate(
        0.0,
        0.0,
        0.0864,
        0.65,
        0.2,
        0.235,
        0.2,
        0.085,
        0.02,
        0.235,
        n_steps,
        0.0,
    )
    trace, spikes, x_final, y_final = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(x_final), float(y_final)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_courage_nekorkin_map_simulate

    assert function.__name__ == "py_courage_nekorkin_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(x0, y0, m0, m1, a, d, j, beta, eps, x_threshold, n_steps, current)"
    )
    assert engine.py_courage_nekorkin_map_simulate is function
    assert "py_courage_nekorkin_map_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_x, empty_y = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_x, empty_y) == (0, 0.0, 0.0)

    one_trace, one_spikes, one_x, one_y = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([0.0], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_x, one_y) == (0, 0.0, -0.004)

    two_trace, two_spikes, two_x, two_y = _direct(2)
    np.testing.assert_array_equal(two_trace, np.array([0.0, 0.004], dtype=np.float64))
    assert (two_spikes, two_x, two_y) == (0, 0.004, -0.008)


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
        extension.py_courage_nekorkin_map_simulate(
            0.0,
            0.0,
            0.0864,
            0.65,
            0.2,
            0.235,
            0.2,
            0.085,
            0.02,
            0.235,
            n_steps,
            0.0,
        )
    assert str(captured.value) == message
    assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert courage_nekorkin_map._HAS_RUST is True
    assert courage_nekorkin_map._rust_simulate is engine.py_courage_nekorkin_map_simulate

    rust_neuron = CourageNekorkinMapNeuron()
    python_neuron = CourageNekorkinMapNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 0.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 0.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.x, rust_neuron.y) == (python_neuron.x, python_neuron.y)
