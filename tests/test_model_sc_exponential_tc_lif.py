# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC exponential two-compartment LIF contracts

"""Frozen-anchor, atomicity, and engine custody for the exponential recurrence."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import (
    SCExponentialTwoCompartmentLIFNeuron as PublicSCExponential,
)
from sc_neurocore.neurons.models.sc_exponential_tc_lif import (
    SCExponentialTwoCompartmentLIFNeuron,
)


def test_public_registry_and_defaults() -> None:
    neuron = SCExponentialTwoCompartmentLIFNeuron()
    assert PublicSCExponential is SCExponentialTwoCompartmentLIFNeuron
    assert (neuron.v_s, neuron.v_d) == (0.0, 0.0)
    assert (neuron.tau_s, neuron.tau_d, neuron.kappa, neuron.dt) == (2.0, 20.0, 0.5, 1.0)


def test_frozen_anchors_match_the_pre_change_engine_bit_exactly() -> None:
    """Anchors captured from the pre-2026-08-27 built engine."""

    neuron = SCExponentialTwoCompartmentLIFNeuron()
    assert [neuron.step(0.5, 0.3) for _ in range(10)] == [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    assert neuron.v_s == 0.0
    assert repr(neuron.v_d) == "2.420328258950689"

    long_run = SCExponentialTwoCompartmentLIFNeuron()
    for _ in range(50):
        long_run.step(0.2, 0.1)
    assert long_run.v_s == 0.0
    assert repr(long_run.v_d) == "1.8821082014698403"


def test_production_engine_class_preserves_the_same_recurrence() -> None:
    engine = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    rust = engine.SCExponentialTwoCompartmentLIF()
    python = SCExponentialTwoCompartmentLIFNeuron()
    for index in range(200):
        i_soma = 0.3 + 0.2 * ((index % 5) - 2)
        i_dend = 0.1 * (index % 3)
        assert rust.step(i_soma, i_dend) == python.step(i_soma, i_dend)
    state = rust.get_state()
    assert state["v_s"] == pytest.approx(python.v_s, abs=1e-12)
    assert state["v_d"] == pytest.approx(python.v_d, abs=1e-12)


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_non_finite_currents_are_rejected_atomically(bad: float) -> None:
    neuron = SCExponentialTwoCompartmentLIFNeuron()
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="i_soma"):
        neuron.step(bad)
    with pytest.raises(ValueError, match="i_soma"):
        neuron.step(0.0, bad)
    assert (neuron.v_s, neuron.v_d) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v_d", math.inf),
        ("v_s", 1.1e6),
        ("v_rest", 100.1),
        ("theta", 100.1),
        ("tau_s", 0.05),
        ("kappa", 10.1),
        ("dt", 10.1),
    ],
)
def test_constructor_rejects_invalid_configuration(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        SCExponentialTwoCompartmentLIFNeuron(**{field: value})


def test_overflowing_candidate_is_rejected_atomically() -> None:
    neuron = SCExponentialTwoCompartmentLIFNeuron()
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.7e308, 1.7e308)
    assert (neuron.v_s, neuron.v_d) == before


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = SCExponentialTwoCompartmentLIFNeuron()
    neuron.dt = 0.0
    before = (neuron.v_s, neuron.v_d)
    with pytest.raises(ValueError, match="dt"):
        neuron.step(0.5)
    assert (neuron.v_s, neuron.v_d) == before


def test_reset_restores_rest_potentials() -> None:
    neuron = SCExponentialTwoCompartmentLIFNeuron(v_rest=0.2)
    neuron.step(0.5, 0.3)
    neuron.reset()
    assert neuron.v_s == 0.2
    assert neuron.v_d == 0.2
