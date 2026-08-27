# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EscapeRate engine-binding contracts

"""Installed-extension contracts for the seeded EscapeRate binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.accel import escape_rate as backends
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters(*, rho_0: float = 0.001, seed: int = 0xACE1) -> tuple[float | int, ...]:
    return (-70.0, -70.0, -70.0, -50.0, 10.0, rho_0, 3.0, 1.0, 1.0, seed)


def _direct(
    n_steps: int, current: float, *, rho_0: float = 0.001, seed: int = 0xACE1
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.uint8]], float, int]:
    trace, events, final_v, final_rng = extension.py_escape_rate_simulate(
        *_parameters(rho_0=rho_0, seed=seed), n_steps, current
    )
    return (
        np.asarray(trace, dtype=np.float64),
        np.asarray(events, dtype=np.uint8),
        float(final_v),
        int(final_rng),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_escape_rate_simulate

    assert function.__name__ == "py_escape_rate_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(v0, v_rest, v_reset, v_threshold, tau_m, rho_0, delta_u, resistance, "
        "dt, rng_state, n_steps, current)"
    )
    assert engine.py_escape_rate_simulate is function
    assert "py_escape_rate_simulate" in engine.__all__


def test_seeded_empty_and_spiking_contracts_are_exact() -> None:
    empty_trace, empty_events, empty_v, empty_rng = _direct(0, 0.0)
    assert empty_trace.shape == empty_events.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_events.dtype == np.uint8
    assert empty_trace.flags.c_contiguous and empty_events.flags.c_contiguous
    assert (empty_v, empty_rng) == (-70.0, 0xACE1)

    trace, events, final_v, final_rng = extension.py_escape_rate_simulate(
        -50.0,
        -70.0,
        -70.0,
        -50.0,
        10.0,
        1.0,
        3.0,
        1.0,
        1.0,
        42,
        8,
        0.0,
    )
    np.testing.assert_array_equal(np.asarray(trace), np.full(8, -70.0))
    np.testing.assert_array_equal(np.asarray(events), np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    assert (final_v, final_rng) == (-70.0, 44136)


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
        extension.py_escape_rate_simulate(*_parameters(), n_steps, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((np.nan, *_parameters()[1:], 1, 0.0), "invalid EscapeRate simulation state or input"),
        ((*_parameters(), 1, np.nan), "invalid EscapeRate simulation state or input"),
        (
            (-70.0, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0e308, 1.0, 42, 1, 1.0e308),
            "non-finite escape-rate membrane candidate",
        ),
    ),
)
def test_invalid_contracts_keep_the_exact_public_error(
    arguments: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        extension.py_escape_rate_simulate(*arguments)


def test_production_rust_backend_is_installed_and_matches_python() -> None:
    assert backends._HAS_RUST is True
    assert backends._engine_simulate is engine.py_escape_rate_simulate

    for current in (0.0, 30.0):
        rust_neuron = EscapeRateNeuron(v=-63.0, seed=0xACE1)
        python_neuron = EscapeRateNeuron(v=-63.0, seed=0xACE1)
        rust_trace, rust_spikes = rust_neuron.simulate(400, current, backend="rust")
        python_trace, python_spikes = python_neuron.simulate(400, current, backend="python")

        np.testing.assert_array_equal(rust_trace, python_trace)
        assert rust_spikes == python_spikes
        assert (rust_neuron.v, rust_neuron.rng_state) == (
            python_neuron.v,
            python_neuron.rng_state,
        )
