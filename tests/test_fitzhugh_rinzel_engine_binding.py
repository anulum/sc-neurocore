# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Rinzel engine-binding contracts

"""Installed-extension contracts for the FitzHugh-Rinzel batch binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import fitzhugh_rinzel
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters() -> tuple[float, ...]:
    neuron = FitzHughRinzelNeuron()
    return (
        neuron.v,
        neuron.w,
        neuron.y,
        neuron.a,
        neuron.b,
        neuron.c,
        neuron.d,
        neuron.delta,
        neuron.mu,
        neuron.dt,
        neuron.v_threshold,
    )


def _direct(
    n_steps: int, current: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int, float, float, float]:
    trace, spikes, final_v, final_w, final_y = extension.py_fitzhugh_rinzel_simulate(
        *_parameters(), n_steps, current
    )
    return np.asarray(trace), int(spikes), float(final_v), float(final_w), float(final_y)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_fitzhugh_rinzel_simulate

    assert function.__name__ == "py_fitzhugh_rinzel_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, w0, y0, a, b, c, d, delta, mu, dt, v_threshold, n_steps, current)"
    )
    assert engine.py_fitzhugh_rinzel_simulate is function
    assert "py_fitzhugh_rinzel_simulate" in engine.__all__


def test_empty_and_early_trajectory_contracts_are_exact() -> None:
    empty = _direct(0, 0.5)
    assert empty[0].shape == (0,)
    assert empty[0].dtype == np.float64
    assert empty[0].flags.c_contiguous
    assert empty[1:] == (0, -1.0, -0.5, 0.0)

    actual = _direct(8, 0.5)
    np.testing.assert_array_equal(
        actual[0],
        [
            -0.9666742099700132,
            -0.9332360397181595,
            -0.8994965998678429,
            -0.8652665899950417,
            -0.8303521568108352,
            -0.7945508410061909,
            -0.7576474619959117,
            -0.7194098099918487,
        ],
    )
    assert actual[1:] == (
        0,
        -0.7194098099918487,
        -0.4851660320121338,
        7.07356631606605e-06,
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
        extension.py_fitzhugh_rinzel_simulate(*_parameters(), n_steps, 0.5)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_invalid_direct_input_preserves_state() -> None:
    trace, spikes, final_v, final_w, final_y = _direct(4, float("nan"))

    np.testing.assert_array_equal(trace, np.full(4, -1.0))
    assert (spikes, final_v, final_w, final_y) == (0, -1.0, -0.5, 0.0)


def test_production_rust_dispatcher_is_installed_and_bit_exact() -> None:
    assert fitzhugh_rinzel._HAS_RUST is True
    assert fitzhugh_rinzel._rust_simulate is engine.py_fitzhugh_rinzel_simulate

    rust_neuron = FitzHughRinzelNeuron()
    python_neuron = FitzHughRinzelNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(8_000, 0.5, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(8_000, 0.5, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.v, rust_neuron.w, rust_neuron.y) == (
        python_neuron.v,
        python_neuron.w,
        python_neuron.y,
    )
