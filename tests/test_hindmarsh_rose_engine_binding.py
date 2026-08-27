# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose engine-binding contracts

"""Installed-extension contracts for the Hindmarsh-Rose batch binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import hindmarsh_rose
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = HindmarshRoseNeuron()
    return (
        neuron.x,
        neuron.y,
        neuron.z,
        neuron.b,
        neuron.r,
        neuron.s,
        neuron.x_rest,
        neuron.dt,
        neuron.x_threshold,
    )


def _direct(
    n_steps: int, current: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, float, float, float]:
    trace, spikes, final_x, final_y, final_z = extension.py_hindmarsh_rose_simulate(
        *_parameters(), n_steps, current
    )
    return np.asarray(trace), int(spikes), float(final_x), float(final_y), float(final_z)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_hindmarsh_rose_simulate

    assert function.__name__ == "py_hindmarsh_rose_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(x0, y0, z0, b, r, s, x_rest, dt, x_threshold, n_steps, current)"
    )
    assert engine.py_hindmarsh_rose_simulate is function
    assert "py_hindmarsh_rose_simulate" in engine.__all__


def test_empty_and_early_trajectory_contracts_are_exact() -> None:
    empty = _direct(0, 3.0)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[0].flags.c_contiguous
    assert empty[1:] == (0, -1.6, -10.0, 2.0)

    actual = _direct(8, 3.0)
    np.testing.assert_array_equal(
        actual[0],
        [
            -1.4770193566363319,
            -1.4407413769305852,
            -1.4270831759180491,
            -1.4194958496603656,
            -1.4135380342625976,
            -1.4080098867588877,
            -1.4025861288315622,
            -1.397177964516537,
        ],
    )
    assert actual[1:] == (0, -1.397177964516537, -9.530718760652906, 1.9989426935281587)


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
        extension.py_hindmarsh_rose_simulate(*_parameters(), n_steps, 3.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_invalid_direct_input_preserves_state() -> None:
    trace, spikes, final_x, final_y, final_z = _direct(4, float("nan"))

    np.testing.assert_array_equal(trace, np.full(4, -1.6))
    assert (spikes, final_x, final_y, final_z) == (0, -1.6, -10.0, 2.0)


def test_production_rust_dispatcher_is_installed_and_bit_exact() -> None:
    assert hindmarsh_rose._HAS_RUST is True
    assert hindmarsh_rose._rust_simulate is engine.py_hindmarsh_rose_simulate

    rust_neuron = HindmarshRoseNeuron()
    python_neuron = HindmarshRoseNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(8_000, 3.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(8_000, 3.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.x, rust_neuron.y, rust_neuron.z) == (
        python_neuron.x,
        python_neuron.y,
        python_neuron.z,
    )
