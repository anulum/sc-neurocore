# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (batch_and_reset) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403


def test_long_varied_run_is_finite_and_deterministic() -> None:
    drive = 4.0 + 1.2 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
    first = ResonateAndFireNeuron()
    second = ResonateAndFireNeuron()
    first_trace = [(first.step(value), first.x, first.y) for value in drive]
    second_trace = [(second.step(value), second.x, second.y) for value in drive]
    assert first_trace == second_trace
    assert np.isfinite((first.x, first.y)).all()


def test_python_batch_matches_scalar_complete_trace() -> None:
    drive = 3.0 + np.sin(np.arange(256, dtype=np.float64) * 0.071)
    batch_neuron = ResonateAndFireNeuron(x=0.15, y=-0.2)
    scalar_neuron = ResonateAndFireNeuron(x=0.15, y=-0.2)
    result = batch_neuron.simulate(drive, backend="python")

    x_expected: list[float] = []
    y_expected: list[float] = []
    spikes_expected: list[float] = []
    for value in drive:
        spikes_expected.append(float(scalar_neuron.step(float(value))))
        x_expected.append(scalar_neuron.x)
        y_expected.append(scalar_neuron.y)

    np.testing.assert_array_equal(result["x"], np.asarray(x_expected))
    np.testing.assert_array_equal(result["y"], np.asarray(y_expected))
    np.testing.assert_array_equal(result["spikes"], np.asarray(spikes_expected))
    assert result["spike_count"] == int(sum(spikes_expected))
    assert (batch_neuron.x, batch_neuron.y) == (scalar_neuron.x, scalar_neuron.y)


def test_empty_batch_preserves_initial_state() -> None:
    neuron = ResonateAndFireNeuron(x=0.25, y=-0.5)
    result = neuron.simulate([], backend="python")
    assert np.asarray(result["x"]).shape == (0,)
    assert np.asarray(result["y"]).shape == (0,)
    assert np.asarray(result["spikes"]).shape == (0,)
    assert result["x_final"] == 0.25
    assert result["y_final"] == -0.5
    assert result["spike_count"] == 0
    assert (neuron.x, neuron.y) == (0.25, -0.5)


def test_reset_restores_quiescent_state_and_preserves_parameters() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.5,
        y=-0.25,
        b=-0.5,
        omega=2.0,
        threshold=3.0,
        dt=0.02,
    )
    neuron.reset()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.b, neuron.omega, neuron.threshold, neuron.dt) == (
        -0.5,
        2.0,
        3.0,
        0.02,
    )
