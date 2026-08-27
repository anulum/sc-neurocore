# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC leaky two-compartment LIF contracts

"""Frozen-anchor and atomicity custody for the preserved leaky recurrence."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import (
    SCLeakyTwoCompartmentLIFNeuron as PublicSCLeaky,
)
from sc_neurocore.neurons.models.sc_leaky_tc_lif import SCLeakyTwoCompartmentLIFNeuron


def test_public_registry_and_defaults() -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron()
    assert PublicSCLeaky is SCLeakyTwoCompartmentLIFNeuron
    assert (neuron.v_s, neuron.v_d) == (0.0, 0.0)
    assert (neuron.tau_s, neuron.tau_d, neuron.kappa, neuron.dt) == (2.0, 20.0, 0.5, 1.0)


def test_frozen_anchors_match_the_pre_change_implementation_bit_exactly() -> None:
    """Anchors captured from the pre-2026-08-27 TwoCompartmentLIFNeuron."""

    neuron = SCLeakyTwoCompartmentLIFNeuron()
    assert [neuron.step(0.5, 0.3) for _ in range(10)] == [0] * 10
    assert repr(neuron.v_s) == "0.37239015011526366"
    assert repr(neuron.v_d) == "0.12037891822848633"

    long_run = SCLeakyTwoCompartmentLIFNeuron()
    for _ in range(50):
        long_run.step(0.2, 0.1)
    assert repr(long_run.v_s) == "0.16405603357692106"
    assert repr(long_run.v_d) == "0.09230550247232869"

    custom = SCLeakyTwoCompartmentLIFNeuron(kappa=0.8, tau_s=3.0)
    assert sum(custom.step(0.4, 0.2) for _ in range(30)) == 0
    assert repr(custom.v_s) == "0.2913383280082365"
    assert repr(custom.v_d) == "0.15707224721141247"


def test_two_current_api_and_spike_hard_reset() -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron(theta=0.5)
    fired = 0
    for _ in range(20):
        fired += neuron.step(1.5, 0.5)
        if fired:
            break
    assert fired == 1
    assert neuron.v_s == neuron.v_reset


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_non_finite_currents_are_rejected_atomically(bad: float) -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron()
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="i_soma"):
        neuron.step(bad)
    with pytest.raises(ValueError, match="i_soma"):
        neuron.step(0.0, bad)
    assert (neuron.v_s, neuron.v_d) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v_s", math.nan),
        ("v_d", 1.1e6),
        ("v_rest", 100.1),
        ("v_reset", -100.1),
        ("theta", 0.0),
        ("tau_s", 0.05),
        ("tau_d", 1000.1),
        ("kappa", -0.1),
        ("dt", 0.0),
    ],
)
def test_constructor_rejects_invalid_configuration(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        SCLeakyTwoCompartmentLIFNeuron(**{field: value})


def test_overflowing_candidate_is_rejected_atomically() -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron(tau_d=0.1, dt=10.0)
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(0.0, 1.7e308)
    assert (neuron.v_s, neuron.v_d) == before


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron()
    neuron.tau_s = 0.0
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="tau_s"):
        neuron.step(0.5)
    assert (neuron.v_s, neuron.v_d) == before


def test_reset_restores_rest_potentials() -> None:
    neuron = SCLeakyTwoCompartmentLIFNeuron(v_rest=0.1)
    neuron.step(0.5, 0.3)
    neuron.reset()
    assert neuron.v_s == 0.1
    assert neuron.v_d == 0.1
