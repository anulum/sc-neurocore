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

import numpy as np
import pytest

import sc_neurocore_engine as engine
from sc_neurocore_engine.studio import get_batch_simulate

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_and_batch_function_identities_are_stable() -> None:
    assert extension.NetworkRunner.__name__ == "NetworkRunner"
    assert extension.NetworkRunner.__module__ == ("sc_neurocore_engine.sc_neurocore_engine")
    assert engine.NetworkRunner is extension.NetworkRunner
    assert extension.py_batch_simulate.__text_signature__ == (
        "(model_name, n_steps, current_trace)"
    )
    assert engine.py_batch_simulate is extension.py_batch_simulate
    assert get_batch_simulate() is extension.py_batch_simulate


def test_population_step_returns_contiguous_typed_arrays() -> None:
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
    current = np.full(200, 10.0)
    result = extension.py_batch_simulate("Izhikevich", 500, current)

    assert result["n_steps"] == 200
    assert result["voltages"].shape == (200,)
    assert result["voltages"].dtype == np.float64
    assert result["spikes"].dtype == np.uint64
    assert result["spikes"].tolist() == [3, 28, 74, 120, 166]
    assert np.all(np.isfinite(result["voltages"]))


def test_model_catalogue_and_unsupported_model_failure_are_exposed() -> None:
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
    with pytest.raises(error) as captured:
        extension.py_batch_simulate("Izhikevich", n_steps, np.ones(1))
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]
