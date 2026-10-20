# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network runner engine-binding contracts

"""Installed-extension contracts for heterogeneous network execution."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore_engine.studio import get_batch_simulate

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_and_batch_function_identities_are_stable() -> None:
    """The package and the Studio getter re-export the extension's own class and function."""
    assert extension.NetworkRunner.__name__ == "NetworkRunner"
    assert extension.NetworkRunner.__module__ == ("sc_neurocore_engine.sc_neurocore_engine")
    assert engine.NetworkRunner is extension.NetworkRunner
    assert extension.py_batch_simulate.__text_signature__ == (
        "(model_name, n_steps, current_trace)"
    )
    assert engine.py_batch_simulate is extension.py_batch_simulate
    assert get_batch_simulate() is extension.py_batch_simulate


def test_population_step_returns_contiguous_typed_arrays() -> None:
    """One population step returns C-contiguous uint8 spikes and float64 voltages."""
    runner = extension.NetworkRunner()
    population = runner.add_population("Izhikevich", 4)
    result = runner.step_population(population, np.full(4, 15.0))

    assert population == 0
    assert result["spikes"].shape == (4,)
    assert result["spikes"].dtype == np.uint8
    assert result["voltages"].shape == (4,)
    assert result["voltages"].dtype == np.float64
    assert result["spikes"].flags.c_contiguous
    assert result["voltages"].flags.c_contiguous


def test_csr_projection_and_multi_population_run_use_real_runner() -> None:
    """A dense CSR projection between two populations runs and returns typed per-population rows."""
    runner = extension.NetworkRunner()
    assert runner.add_population("Izhikevich", 3) == 0
    assert runner.add_population("AdEx", 3) == 1
    runner.add_projection(
        0,
        1,
        [0, 3, 6, 9],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0.5] * 9,
    )

    result = runner.run(100)
    assert result["spike_counts"].shape == (2,)
    assert result["spike_counts"].dtype == np.uint64
    assert len(result["spike_data"]) == 2
    assert len(result["voltages"]) == 2
    assert all(row.dtype == np.uint64 for row in result["spike_data"])
    assert all(row.dtype == np.float64 for row in result["voltages"])


def test_batch_simulation_truncates_to_current_trace_and_is_exact() -> None:
    """The batch run stops at the trace length and reproduces the pinned Izhikevich spike steps."""
    current = np.full(200, 10.0)
    result = extension.py_batch_simulate("Izhikevich", 500, current)

    assert result["n_steps"] == 200
    assert result["voltages"].shape == (200,)
    assert result["voltages"].dtype == np.float64
    assert result["spikes"].dtype == np.uint64
    assert result["spikes"].tolist() == [3, 28, 74, 120, 166]
    assert np.all(np.isfinite(result["voltages"]))


def test_model_catalogue_and_unsupported_model_failure_are_exposed() -> None:
    """The catalogue lists the core models and an unknown model raises ValueError."""
    models = extension.NetworkRunner.supported_models()

    assert len(models) >= 160
    assert {
        "Izhikevich",
        "AdEx",
        "HodgkinHuxley",
        "McCullochPittsNeuron",
    } <= set(models)
    with pytest.raises(ValueError, match="Unsupported model: 'NonexistentModel'"):
        extension.py_batch_simulate("NonexistentModel", 1, np.ones(1))


@pytest.mark.parametrize(
    ("n_steps", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_batch_step_count_conversion_errors_are_stable(
    n_steps: object, error: type[BaseException], message: str
) -> None:
    """Negative and non-integer step counts keep their PyO3 conversion errors and notes."""
    with pytest.raises(error) as captured:
        extension.py_batch_simulate("Izhikevich", n_steps, np.ones(1))
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def _driven_pair(delay: int = 0) -> Any:
    """Return two one-neuron McCulloch-Pitts populations, 0 → 1, with source 0 spiking."""
    runner = extension.NetworkRunner()
    runner.add_population("McCullochPittsNeuron", 1)
    runner.add_population("McCullochPittsNeuron", 1)
    runner.add_projection(0, 1, [0, 1], [0], [1.0], delay)
    runner.step_population(0, np.ones(1))
    return runner


@pytest.mark.parametrize(
    ("src", "tgt", "row_offsets", "col_indices", "values", "message"),
    (
        (2, 0, [0, 1], [0], [1.0], "source population index 2 out of range"),
        (0, 9, [0, 1], [0], [1.0], "target population index 9 out of range"),
        (0, 1, [], [], [], "row_offsets has 0 entries"),
        (0, 1, [0, 1, 1], [0], [1.0], "row_offsets has 3 entries"),
        (0, 1, [1, 1], [0], [1.0], "must start at 0"),
        (0, 1, [0, -1], [], [], "must not decrease"),
        (0, 1, [0, 1], [0], [], "col_indices has 1 entries but values has 0"),
        (0, 1, [0, 2], [0], [1.0], "the last row offset 2 exceeds"),
        (0, 1, [0, 1], [1], [1.0], r"col_indices\[0\] = 1 is outside"),
        (0, 1, [0, 1], [-1], [1.0], r"col_indices\[0\] = -1 is outside"),
    ),
)
def test_malformed_csr_is_refused_and_the_runner_is_unchanged(
    src: int,
    tgt: int,
    row_offsets: list[int],
    col_indices: list[int],
    values: list[float],
    message: str,
) -> None:
    """Review 291: no malformed projection is appended, and the healthy one still delivers."""
    runner = _driven_pair()
    with pytest.raises(ValueError, match=message):
        runner.add_projection(src, tgt, row_offsets, col_indices, values, 2)
    assert runner.run(1)["spike_counts"].tolist() == [0, 1]


def test_a_delayed_projection_delivers_after_its_delay() -> None:
    """A delay of two steps delivers the source spike on the third run step only."""
    runner = _driven_pair(delay=2)
    assert [runner.run(1)["spike_counts"][1] for _ in range(4)] == [0, 0, 1, 0]
