# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson engine-binding contracts

"""Installed-extension contracts for the seeded homogeneous Poisson binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.accel import poisson as backends
from sc_neurocore.neurons.models.poisson import PoissonNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _parameters(*, seed: int = 0xACE1) -> tuple[float | int, ...]:
    return (250.0, 1.0, seed)


def _direct(
    n_steps: int, rate_override: float = -1.0, *, seed: int = 0xACE1
) -> tuple[np.ndarray[Any, np.dtype[np.uint8]], int]:
    events, final_rng = extension.py_poisson_simulate(
        *_parameters(seed=seed), n_steps, rate_override
    )
    return np.asarray(events, dtype=np.uint8), int(final_rng)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_poisson_simulate

    assert function.__name__ == "py_poisson_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(rate_hz, dt_ms, rng_state, n_steps, rate_override=...)"
    )
    assert engine.py_poisson_simulate is function
    assert "py_poisson_simulate" in engine.__all__


def test_seeded_empty_and_event_stream_contracts_are_exact() -> None:
    empty_events, empty_rng = _direct(0)
    assert empty_events.shape == (0,)
    assert empty_events.dtype == np.uint8
    assert empty_events.flags.c_contiguous
    assert empty_rng == 0xACE1

    events, final_rng = _direct(8)
    np.testing.assert_array_equal(events, np.array([1, 0, 1, 0, 0, 0, 1, 0]))
    assert final_rng == 34_837


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
        extension.py_poisson_simulate(*_parameters(), n_steps, -1.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((np.nan, 1.0, 0xACE1, 1, -1.0), "invalid Poisson simulation state or rate override"),
        ((250.0, 1.0, 0, 1, -1.0), "invalid Poisson simulation state or rate override"),
        ((250.0, 1.0, 0xACE1, 1, np.nan), "invalid Poisson simulation state or rate override"),
        ((1.0e308, 1.0e308, 0xACE1, 1, -1.0), "non-finite Poisson interval hazard"),
    ),
)
def test_invalid_contracts_keep_the_exact_public_error(
    arguments: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        extension.py_poisson_simulate(*arguments)


@pytest.mark.parametrize("rate_override", (-1.0, 0.0, 500.0))
def test_production_rust_backend_is_installed_and_matches_python(
    rate_override: float,
) -> None:
    assert backends._HAS_RUST is True
    assert backends._engine_simulate is engine.py_poisson_simulate

    rust_neuron = PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0x1234)
    python_neuron = PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0x1234)
    rust_events, rust_count = rust_neuron.simulate(4096, rate_override, backend="rust")
    python_events, python_count = python_neuron.simulate(4096, rate_override, backend="python")

    np.testing.assert_array_equal(rust_events, python_events)
    assert rust_count == python_count
    assert rust_neuron.rng_state == python_neuron.rng_state
