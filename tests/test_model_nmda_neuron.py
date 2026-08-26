# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NMDA source-model contracts

"""Exact source coverage for the WB base + Jahr-Stevens NMDA channel model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import NMDANeuron as PublicNMDANeuron
from sc_neurocore.neurons.models import nmda_neuron as module
from sc_neurocore.neurons.models.nmda_neuron import NMDANeuron


def _state(neuron: NMDANeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.s_nmda


def test_public_registry_defaults_and_nominal_anchor() -> None:
    neuron = NMDANeuron()
    assert PublicNMDANeuron is NMDANeuron
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.0)
    assert neuron._sub_steps == 50
    assert neuron.step(5.0) == 0
    assert _state(neuron) == pytest.approx(
        (-63.155663780395777, 0.6480311943997441, 0.23722188716377601, 0.025),
        abs=1.0e-15,
    )


def test_rate_singularity_branches() -> None:
    assert module._safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0
    assert module._safe_rate(0.1, 35.0, -34.0, 10.0, 1.0) > 0.0


def test_mg_block_follows_jahr_stevens_voltage_dependence() -> None:
    block_rest = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * -65.0))
    block_depolarised = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * -20.0))
    assert block_rest < 0.1
    assert block_depolarised > 0.4


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("v", math.nan, ValueError),
        ("v", -100.1, ValueError),
        ("h", -0.1, ValueError),
        ("n", 1.1, ValueError),
        ("s_nmda", 1.1, ValueError),
        ("g_na", -0.1, ValueError),
        ("g_k", 100.1, ValueError),
        ("g_nmda", 20.1, ValueError),
        ("g_l", 5.1, ValueError),
        ("e_na", 29.9, ValueError),
        ("e_k", -69.9, ValueError),
        ("e_nmda", 10.1, ValueError),
        ("e_l", -39.9, ValueError),
        ("c_m", 0.49, ValueError),
        ("phi", 10.1, ValueError),
        ("mg_conc", 5.1, ValueError),
        ("tau_rise", 0.09, ValueError),
        ("tau_decay", 500.1, ValueError),
        ("dt", 0.0, ValueError),
        ("v_threshold", 20.1, ValueError),
        ("gain", 10.1, ValueError),
        ("_sub_steps", 1.5, TypeError),
        ("_sub_steps", True, TypeError),
        ("_sub_steps", 0, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(
    field: str, value: float | bool, error: type[Exception]
) -> None:
    with pytest.raises(error):
        NMDANeuron(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_atomically(current: float) -> None:
    neuron = NMDANeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_infinite_drive_no_longer_emits_a_false_spike() -> None:
    """Regression for NCDBG-045: step(+inf) used to return 1 and clamp s_nmda to 1."""

    neuron = NMDANeuron()
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.inf)
    assert neuron.s_nmda == 0.0
    assert neuron.v == -65.0


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = NMDANeuron()
    neuron.c_m = 0.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="c_m"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = NMDANeuron()
    before = _state(neuron)
    monkeypatch.setattr(module, "_safe_rate", lambda *_args: math.inf)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_s_nmda_rises_with_input_and_decays_without() -> None:
    neuron = NMDANeuron()
    assert neuron.step(5.0) == 0
    assert neuron.s_nmda == pytest.approx(0.025, abs=1.0e-15)
    for _ in range(200):
        neuron.step(5.0)
    peak = neuron.s_nmda
    assert peak > 0.2
    for _ in range(200):
        neuron.step(0.0)
    assert neuron.s_nmda < peak


def test_s_nmda_saturation_clamps_at_unity() -> None:
    neuron = NMDANeuron(tau_rise=0.1, s_nmda=0.9, gain=10.0)
    neuron.step(9.9)
    assert neuron.s_nmda == 1.0


def test_spike_path_bounds_and_reset_preserves_parameters() -> None:
    neuron = NMDANeuron(g_nmda=1.5, v=-20.0)
    assert neuron.step(1.0e6) == 1
    assert -100.0 <= neuron.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (neuron.h, neuron.n, neuron.s_nmda))
    neuron.reset()
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.0)
    assert neuron.g_nmda == 1.5
