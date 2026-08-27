# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical TC-LIF source-model contracts

"""Exact source coverage for the Zhang et al. (2024) TC-LIF map."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import TwoCompartmentLIFNeuron as PublicTCLIF
from sc_neurocore.neurons.models.tc_lif import TC_LIF_PROFILES, TwoCompartmentLIFNeuron


def _state(neuron: TwoCompartmentLIFNeuron) -> tuple[float, float, float]:
    return neuron.u_d, neuron.u_s, neuron.s_prev


def test_public_registry_and_smnist_feedforward_defaults() -> None:
    neuron = TwoCompartmentLIFNeuron()
    assert PublicTCLIF is TwoCompartmentLIFNeuron
    assert _state(neuron) == (0.0, 0.0, 0.0)
    assert (neuron.beta1, neuron.beta2, neuron.gamma, neuron.v_th) == (-0.5, 0.5, 0.5, 1.0)


def test_matches_independent_paper_oracle_bit_exactly() -> None:
    """Eq.10 -> Eq.11 -> Eq.12 ordering with the delayed S[t-1] reset."""

    beta1, beta2, gamma, v_th = -0.5, 0.5, 0.5, 1.0
    u_d = u_s = s_prev = 0.0
    neuron = TwoCompartmentLIFNeuron()
    for index in range(300):
        current = 0.3 + 0.2 * ((index % 7) - 3)
        u_d = u_d + beta1 * u_s + current - gamma * s_prev
        u_s = u_s + beta2 * u_d - v_th * s_prev
        s_prev = 1.0 if u_s >= v_th else 0.0
        assert neuron.step(current) == int(s_prev)
        assert _state(neuron) == (u_d, u_s, s_prev)


def test_delayed_soft_reset_subtracts_gamma_and_v_th_on_the_next_step() -> None:
    neuron = TwoCompartmentLIFNeuron()
    assert neuron.step(4.0) == 1
    u_d, u_s = neuron.u_d, neuron.u_s
    neuron.step(0.0)
    expected_u_d = u_d + neuron.beta1 * u_s + 0.0 - neuron.gamma
    expected_u_s = u_s + neuron.beta2 * expected_u_d - neuron.v_th
    assert neuron.u_d == expected_u_d
    assert neuron.u_s == expected_u_s


def test_every_published_table5_profile_constructs_and_steps() -> None:
    assert len(TC_LIF_PROFILES) == 10
    for name, (beta1, beta2, gamma, v_th) in TC_LIF_PROFILES.items():
        neuron = TwoCompartmentLIFNeuron.from_profile(name)
        assert (neuron.beta1, neuron.beta2, neuron.gamma, neuron.v_th) == (
            beta1,
            beta2,
            gamma,
            v_th,
        )
        assert -1.0 < beta1 < 0.0 and 0.0 < beta2 < 1.0
        assert neuron.step(0.5) in (0, 1)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown TC-LIF profile"):
        TwoCompartmentLIFNeuron.from_profile("does_not_exist")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("u_d", math.nan),
        ("u_d", 1.1e6),
        ("u_s", -1.1e6),
        ("s_prev", 0.5),
        ("beta1", 0.5),
        ("beta1", -1.0),
        ("beta2", -0.5),
        ("beta2", 1.0),
        ("gamma", -0.1),
        ("gamma", 10.1),
        ("v_th", 0.0),
        ("v_th", 100.1),
    ],
)
def test_constructor_rejects_invalid_configuration(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        TwoCompartmentLIFNeuron(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_input_is_rejected_atomically(current: float) -> None:
    neuron = TwoCompartmentLIFNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="i_ext"):
        neuron.step(current)
    assert _state(neuron) == before


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = TwoCompartmentLIFNeuron()
    neuron.beta2 = 2.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="beta2"):
        neuron.step(0.0)
    assert _state(neuron) == before


def test_extreme_finite_input_stays_finite_and_is_then_bounded_out() -> None:
    """Binary64 absorption keeps candidates finite; the state bound rejects next."""

    neuron = TwoCompartmentLIFNeuron(u_d=1e6)
    assert neuron.step(1.7976931348623157e308) == 1
    assert math.isfinite(neuron.u_d) and math.isfinite(neuron.u_s)
    with pytest.raises(ValueError, match="u_d"):
        neuron.step(0.0)


def test_runaway_state_fails_closed_on_a_later_step() -> None:
    neuron = TwoCompartmentLIFNeuron()
    neuron.u_d = 9.9e5
    with pytest.raises(ValueError, match="u_d|u_s"):
        for _ in range(200):
            neuron.step(1e5)


def test_reset_preserves_parameters() -> None:
    neuron = TwoCompartmentLIFNeuron.from_profile("psmnist_recurrent")
    neuron.step(2.0)
    neuron.reset()
    assert _state(neuron) == (0.0, 0.0, 0.0)
    assert (neuron.beta1, neuron.beta2) == (-0.2, 0.8)
