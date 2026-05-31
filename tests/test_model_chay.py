# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ChayNeuron behavioural contract tests

"""Module-specific Chay 1985 pancreatic beta-cell dynamics contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.chay import ChayNeuron


def _snapshot(neuron: ChayNeuron) -> tuple[float, float, float]:
    return neuron.v, neuron.n, neuron.ca


def test_default_step_preserves_finite_biophysical_state() -> None:
    neuron = ChayNeuron()

    spikes = [neuron.step(0.0) for _ in range(200)]

    assert set(spikes) <= {0, 1}
    assert -200.0 <= neuron.v <= 200.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.ca <= neuron._CA_MAX


def test_internal_substeps_stabilize_stiff_default_timestep() -> None:
    default_dt = ChayNeuron(dt=0.02)
    small_dt = ChayNeuron(dt=0.001)

    default_dt.step(0.0)
    for _ in range(20):
        small_dt.step(0.0)

    assert default_dt.v == pytest.approx(small_dt.v, abs=1e-9)
    assert default_dt.n == pytest.approx(small_dt.n, abs=1e-9)
    assert default_dt.ca == pytest.approx(small_dt.ca, abs=1e-12)


def test_potassium_conductance_hyperpolarizes_relative_to_blocked_k() -> None:
    blocked = ChayNeuron(g_k=0.0, g_kca=0.0)
    intact = ChayNeuron()

    for _ in range(50):
        blocked.step(0.0)
        intact.step(0.0)

    assert intact.v < blocked.v


def test_external_current_depolarizes_relative_to_rest() -> None:
    rest = ChayNeuron()
    driven = ChayNeuron()

    for _ in range(50):
        rest.step(0.0)
        driven.step(1000.0)

    assert driven.v > rest.v


def test_reset_restores_dynamic_state_without_rewriting_parameters() -> None:
    neuron = ChayNeuron(dt=0.01, g_k=1200.0)
    for _ in range(20):
        neuron.step(100.0)

    neuron.reset()

    assert _snapshot(neuron) == (-50.0, 0.1, 0.1)
    assert neuron.dt == 0.01
    assert neuron.g_k == 1200.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", 250.0),
        ("n", -0.01),
        ("n", 1.01),
        ("ca", -1e-6),
        ("ca", 101.0),
        ("g_ca", -1.0),
        ("g_k", -1.0),
        ("g_kca", -1.0),
        ("g_l", -1.0),
        ("rho", -1e-6),
        ("alpha_ca", -1e-6),
        ("k_ca", -1e-6),
        ("dt", 0.0),
        ("dt", float("inf")),
    ],
)
def test_invalid_runtime_or_parameter_state_does_not_mutate(field: str, value: float) -> None:
    neuron = ChayNeuron()
    setattr(neuron, field, value)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.0)

    assert _snapshot(neuron) == before


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = ChayNeuron()
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(float("nan"))

    assert _snapshot(neuron) == before


def test_candidate_outside_voltage_envelope_does_not_mutate_state() -> None:
    neuron = ChayNeuron(v=-190.0, n=1.0, dt=0.5)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(-1e6)

    assert _snapshot(neuron) == before


def test_candidate_outside_calcium_envelope_does_not_mutate_state() -> None:
    neuron = ChayNeuron(rho=1e9, dt=0.001)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.0)

    assert _snapshot(neuron) == before
