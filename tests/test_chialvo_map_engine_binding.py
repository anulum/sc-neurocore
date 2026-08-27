# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chialvo-map engine-binding contracts

"""Installed-extension contracts for the Chialvo-map binding."""

from __future__ import annotations

import importlib
import math
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import chialvo_map
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_chialvo_map_simulate(
        0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, n_steps, 0.0
    )
    trace, spikes, x_final, y_final = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(x_final), float(y_final)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_chialvo_map_simulate

    assert function.__name__ == "py_chialvo_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == ("(x0, y0, a, b, c, k, x_threshold, n_steps, current)")
    assert engine.py_chialvo_map_simulate is function
    assert "py_chialvo_map_simulate" in engine.__all__


def test_empty_and_single_step_results_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_x, empty_y = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_x, empty_y) == (0, 0.0, 0.0)

    trace, spikes, x_final, y_final = _direct(1)
    np.testing.assert_array_equal(trace, np.array([0.04], dtype=np.float64))
    assert trace.flags.c_contiguous
    assert (spikes, x_final, y_final) == (0, 0.04, 0.28)


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
        extension.py_chialvo_map_simulate(0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, n_steps, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((math.nan, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, 1, 0.0), "invalid Chialvo map runtime state"),
        ((0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, 1, math.inf), "invalid Chialvo map current"),
        (
            (1e308, 1e308, 0.89, 0.6, 0.28, 0.04, 1.0, 1, 0.0),
            "invalid Chialvo map candidate state",
        ),
    ),
)
def test_checked_errors_remain_floating_point_errors(
    arguments: tuple[float, ...], message: str
) -> None:
    with pytest.raises(FloatingPointError, match=f"^{message}$"):
        extension.py_chialvo_map_simulate(*arguments)


def test_production_rust_backend_uses_the_installed_extension() -> None:
    assert chialvo_map._HAS_RUST is True
    assert chialvo_map._rust_simulate is engine.py_chialvo_map_simulate

    rust_neuron = ChialvoMapNeuron()
    python_neuron = ChialvoMapNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(128, 0.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(128, 0.0, backend="python")

    np.testing.assert_allclose(
        rust_trace,
        python_trace,
        rtol=float(4 * np.finfo(np.float64).eps),
        atol=0.0,
    )
    assert rust_spikes == python_spikes
    assert (rust_neuron.x, rust_neuron.y) == (python_neuron.x, python_neuron.y)
