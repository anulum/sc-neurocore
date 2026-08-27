# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NMDA source and retained-identity contracts

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from sc_neurocore.neurons.models import (
    NMDANeuron as PublicNMDANeuron,
    SCWBNMDAMagnesiumBlockNeuron as PublicSCNeuron,
)
from sc_neurocore.neurons.models.nmda_neuron import NMDANeuron
from sc_neurocore.neurons.models.sc_wb_nmda_magnesium_block import (
    SCWBNMDAMagnesiumBlockNeuron,
)


def _source_state(neuron: NMDANeuron) -> tuple[float, float, float, float, float]:
    return neuron.v, neuron.x_nmda, neuron.s_nmda, neuron.ca, neuron.refractory_remaining


def _sc_state(neuron: SCWBNMDAMagnesiumBlockNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.s_nmda


def test_public_registry_and_source_defaults() -> None:
    neuron = NMDANeuron()
    assert PublicNMDANeuron is NMDANeuron
    assert _source_state(neuron) == (-70.0, 0.0, 0.0, 0.0, 0.0)
    assert (neuron.c_m, neuron.g_l, neuron.v_threshold, neuron.v_reset) == (
        0.5,
        0.025,
        -52.0,
        -59.0,
    )


def test_source_one_step_anchor() -> None:
    neuron = NMDANeuron()
    assert neuron.step(0.3) == 0
    assert _source_state(neuron) == pytest.approx((-69.9700375, 0.0, 0.0, 0.0, 0.0), abs=1.0e-15)


def test_source_midpoint_rk2_differs_from_euler() -> None:
    neuron = NMDANeuron(v=-60.0, x_nmda=0.7, s_nmda=0.25, ca=0.3, g_ahp=0.08)
    initial = _source_state(neuron)
    derivative = neuron._derivatives(*initial[:4], 0.9)
    euler_v = initial[0] + neuron.dt * derivative[0]
    neuron.step(0.9)
    assert neuron.v != pytest.approx(euler_v, abs=1.0e-12)


def test_source_autapse_is_driven_only_by_emitted_events() -> None:
    silent = NMDANeuron()
    for _ in range(100):
        silent.step(0.0)
    assert (silent.x_nmda, silent.s_nmda) == (0.0, 0.0)

    firing = NMDANeuron(v=-52.01)
    assert firing.step(1.0) == 1
    assert firing.x_nmda > 0.0
    assert firing.ca > 0.0


def test_source_refractory_hold_preserves_gate_dynamics() -> None:
    neuron = NMDANeuron(v=-52.01)
    assert neuron.step(1.0) == 1
    x_before = neuron.x_nmda
    assert neuron.step(0.0) == 0
    assert neuron.v == neuron.v_reset
    assert neuron.refractory_remaining == pytest.approx(1.95)
    assert neuron.x_nmda < x_before
    assert neuron.s_nmda > 0.0


def test_magnesium_block_is_voltage_dependent() -> None:
    rest = 1.0 / (1.0 + math.exp(-0.062 * -70.0) / 3.57)
    depolarised = 1.0 / (1.0 + math.exp(-0.062 * -20.0) / 3.57)
    assert rest < 0.05
    assert depolarised > 0.5


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("v", math.nan, ValueError),
        ("v", -120.1, ValueError),
        ("x_nmda", -0.1, ValueError),
        ("s_nmda", 1.1, ValueError),
        ("ca", -0.1, ValueError),
        ("c_m", 0.0, ValueError),
        ("g_nmda", 2.1, ValueError),
        ("mg_conc", 5.1, ValueError),
        ("tau_x", 0.0, ValueError),
        ("tau_s", 0.5, ValueError),
        ("kinetic_scale", 0.0, ValueError),
        ("g_ahp", -0.1, ValueError),
        ("tau_ca", 0.5, ValueError),
        ("dt", 0.051, ValueError),
        ("v_threshold", -29.9, ValueError),
        ("v_reset", -51.0, ValueError),
        ("refractory_period", -0.1, ValueError),
    ],
)
def test_source_constructor_rejects_invalid_configuration(
    field: str, value: float, error: type[Exception]
) -> None:
    constructor: Callable[..., NMDANeuron] = NMDANeuron
    with pytest.raises(error):
        constructor(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_source_drive_is_rejected_atomically(current: float) -> None:
    neuron = NMDANeuron()
    before = _source_state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _source_state(neuron) == before


def test_corrupted_source_configuration_is_rejected_atomically() -> None:
    neuron = NMDANeuron()
    neuron.c_m = 0.0
    before = _source_state(neuron)
    with pytest.raises(ValueError, match="c_m"):
        neuron.step(1.0)
    assert _source_state(neuron) == before


def test_source_reset_preserves_parameters() -> None:
    neuron = NMDANeuron(g_nmda=0.7, v=-52.01)
    neuron.step(1.0)
    neuron.reset()
    assert _source_state(neuron) == (-70.0, 0.0, 0.0, 0.0, 0.0)
    assert neuron.g_nmda == 0.7


def test_retained_sc_identity_and_historical_anchor() -> None:
    neuron = SCWBNMDAMagnesiumBlockNeuron()
    assert PublicSCNeuron is SCWBNMDAMagnesiumBlockNeuron
    assert neuron.step(5.0) == 0
    assert _sc_state(neuron) == pytest.approx(
        (-63.15566378039578, 0.6480311943997441, 0.237221887163776, 0.025),
        abs=1.0e-14,
    )


def test_retained_sc_identity_is_distinct_from_source() -> None:
    assert NMDANeuron is not SCWBNMDAMagnesiumBlockNeuron
    source = NMDANeuron()
    retained = SCWBNMDAMagnesiumBlockNeuron()
    source.step(1.0)
    retained.step(1.0)
    assert source.v != retained.v


def test_retained_sc_invalid_drive_is_atomic() -> None:
    neuron = SCWBNMDAMagnesiumBlockNeuron()
    before = _sc_state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert _sc_state(neuron) == before
