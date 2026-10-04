# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Configured source MAT five-runtime contracts

"""Compare complete configured source MAT trajectories on actual runtimes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import cast

import numpy as np
import pytest

from sc_neurocore.accel.mat import PARITY_ATOL, backend_available, simulate_mat
from sc_neurocore.neurons.models.mat import MATNeuron

PROFILES = [
    {},
    {"omega": 26.0, "alpha_1": 1.7, "alpha_2": 2.0},
    {"omega": 11.0, "alpha_1": 10.0, "alpha_2": 0.002},
    {
        "v": 25.0,
        "theta1": 2.0,
        "theta2": 3.0,
        "refractory_remaining": 0.1,
        "omega": 14.0,
        "tau_m": 8.0,
        "tau_1": 12.0,
        "tau_2": 180.0,
        "alpha_1": 7.0,
        "alpha_2": 1.5,
        "resistance": 40.0,
        "refractory_period": 1.0,
        "dt": 0.05,
    },
    {"dt": 0.01},
    {"dt": 0.1},
    {"dt": 0.5},
    {"refractory_period": 0.0, "v": 30.0, "omega": 15.0},
    {"v": -200.0, "theta1": 1e9, "theta2": 1e9, "alpha_1": 0.0, "alpha_2": 0.0},
]


@pytest.mark.parametrize("parameters", PROFILES)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_complete_profile_preserves_full_trace_and_empty_finals(
    parameters: dict[str, float], backend: str
) -> None:
    """Match every state, exact event, immutable input and configured empty final."""
    assert backend_available(backend), f"required MAT backend {backend} unavailable"
    drive = np.resize(np.array([0.7, 0.0, 0.5, 0.2], dtype=np.float64), 256)
    before = drive.copy()
    drive.setflags(write=False)
    expected = simulate_mat(drive, backend="python", **parameters)
    actual = simulate_mat(drive, backend=backend, **parameters)
    np.testing.assert_array_equal(actual["events"], expected["events"])
    assert np.asarray(actual["events"]).dtype == np.int64
    for key in ("voltages", "theta1", "theta2", "refractory"):
        output = np.asarray(actual[key])
        assert output.dtype == np.float64 and output.shape == (256,)
        np.testing.assert_allclose(output, expected[key], rtol=0, atol=PARITY_ATOL[backend])
    for key in ("v_final", "theta1_final", "theta2_final", "refractory_final"):
        assert float(cast(float, actual[key])) == pytest.approx(
            expected[key], rel=0, abs=PARITY_ATOL[backend]
        )
    empty = simulate_mat([], backend=backend, **parameters)
    initial = simulate_mat([], backend="python", **parameters)
    for key in ("voltages", "theta1", "theta2", "refractory", "events"):
        assert np.asarray(empty[key]).shape == (0,)
    for key in ("v_final", "theta1_final", "theta2_final", "refractory_final"):
        assert empty[key] == initial[key]
    np.testing.assert_array_equal(drive, before)
    assert not drive.flags.writeable


@pytest.mark.parametrize(
    "factory", [MATNeuron.regular_spiking, MATNeuron.intrinsically_bursting, MATNeuron.fast_spiking]
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", 4.0),
        ("theta1", 1.0),
        ("theta2", 2.0),
        ("refractory_remaining", 0.5),
        ("omega", 15.0),
        ("tau_m", 8.0),
        ("tau_1", 12.0),
        ("tau_2", 250.0),
        ("alpha_1", 7.0),
        ("alpha_2", 3.0),
        ("resistance", 40.0),
        ("refractory_period", 1.0),
        ("dt", 0.1),
    ],
)
@pytest.mark.parametrize("backend", ["python", "rust", "julia", "go", "mojo"])
def test_factory_override_reaches_complete_actual_runtime(
    factory: Callable[..., MATNeuron], field: str, value: float, backend: str
) -> None:
    """Named-profile overrides preserve all dynamics and events across actual runtimes."""
    assert backend_available(backend), f"required MAT backend {backend} unavailable"
    overrides = {"v": 40.0, "theta1": 2.0, "theta2": 3.0, "dt": 0.05}
    overrides[field] = value
    cell = factory(**overrides)
    configuration = {item.name: float(getattr(cell, item.name)) for item in fields(MATNeuron)}
    drive = np.resize(np.array([0.7, 0.0, 0.5, 0.2], dtype=np.float64), 256)
    before = drive.copy()
    drive.setflags(write=False)
    trace = []
    for sample in drive:
        event = cell.step(float(sample))
        trace.append((cell.v, cell.theta1, cell.theta2, cell.refractory_remaining, event))
    actual = simulate_mat(drive, backend=backend, **configuration)
    for index, key in enumerate(("voltages", "theta1", "theta2", "refractory", "events")):
        output = np.asarray(actual[key])
        assert output.shape == (256,)
        assert output.dtype == (np.int64 if key == "events" else np.float64)
        if key == "events":
            np.testing.assert_array_equal(output, [row[index] for row in trace])
        else:
            np.testing.assert_allclose(
                output, [row[index] for row in trace], rtol=0, atol=PARITY_ATOL[backend]
            )
    for key, value in (
        ("v_final", cell.v),
        ("theta1_final", cell.theta1),
        ("theta2_final", cell.theta2),
        ("refractory_final", cell.refractory_remaining),
    ):
        assert actual[key] == pytest.approx(value, rel=0, abs=PARITY_ATOL[backend])
    empty = simulate_mat([], backend=backend, **configuration)
    assert tuple(
        empty[key] for key in ("v_final", "theta1_final", "theta2_final", "refractory_final")
    ) == tuple(configuration[key] for key in ("v", "theta1", "theta2", "refractory_remaining"))
    np.testing.assert_array_equal(drive, before)
    assert not drive.flags.writeable
