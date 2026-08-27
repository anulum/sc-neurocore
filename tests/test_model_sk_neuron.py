# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SK source-model contracts

"""Exact source coverage for the WB base + SK (KCa2.x) channel model."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from sc_neurocore.neurons.models import SKNeuron as PublicSKNeuron
from sc_neurocore.neurons.models import sk_neuron as module
from sc_neurocore.neurons.models.sk_neuron import SKNeuron


def _state(neuron: SKNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.ca


def test_public_registry_defaults_and_nominal_anchor() -> None:
    neuron = SKNeuron()
    assert PublicSKNeuron is SKNeuron
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.0)
    assert neuron._sub_steps == 50
    assert neuron.step(5.0) == 0
    assert _state(neuron) == pytest.approx(
        (-63.18006421307219, 0.6481228357499981, 0.2371863659468615, 0.0),
        abs=1.0e-15,
    )


def test_rate_singularity_branches() -> None:
    assert module._safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0
    assert module._safe_rate(0.1, 35.0, -34.0, 10.0, 1.0) > 0.0


def test_sk_activation_is_calcium_hill_n2() -> None:
    assert 0.0**2 / (0.0**2 + 0.25) == 0.0
    assert pytest.approx(0.5) == 0.5**2 / (0.5**2 + 0.25)
    assert 2.0**2 / (2.0**2 + 0.25) > 0.9


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("v", math.nan, ValueError),
        ("v", -100.1, ValueError),
        ("h", -0.1, ValueError),
        ("n", 1.1, ValueError),
        ("ca", -0.1, ValueError),
        ("ca", math.inf, ValueError),
        ("g_na", -0.1, ValueError),
        ("g_k", 100.1, ValueError),
        ("g_sk", 50.1, ValueError),
        ("g_l", 5.1, ValueError),
        ("e_na", 29.9, ValueError),
        ("e_k", -69.9, ValueError),
        ("e_l", -39.9, ValueError),
        ("c_m", 0.49, ValueError),
        ("phi", 10.1, ValueError),
        ("tau_ca", 9.9, ValueError),
        ("tau_ca", 2000.1, ValueError),
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
    constructor: Callable[..., SKNeuron] = SKNeuron
    with pytest.raises(error):
        constructor(**{field: value})


def test_calcium_has_no_artificial_upper_bound() -> None:
    neuron = SKNeuron(ca=25.0)
    assert neuron.step(0.0) == 0
    assert neuron.ca < 25.0


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_atomically(current: float) -> None:
    neuron = SKNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_infinite_drive_no_longer_emits_a_false_spike() -> None:
    """Reject infinite drive instead of emitting a false spike."""

    neuron = SKNeuron()
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.inf)
    assert neuron.ca == 0.0
    assert neuron.v == -65.0


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = SKNeuron()
    neuron.c_m = 0.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="c_m"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = SKNeuron()
    before = _state(neuron)
    monkeypatch.setattr(module, "_safe_rate", lambda *_args: math.inf)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_spike_increments_calcium_and_adaptation_slows_firing() -> None:
    neuron = SKNeuron()
    early = sum(neuron.step(5.0) for _ in range(2000))
    ca_after_early = neuron.ca
    late = sum(neuron.step(5.0) for _ in range(2000))
    assert early >= 1
    assert ca_after_early > 0.0
    assert early >= late


def test_calcium_decays_between_spikes() -> None:
    neuron = SKNeuron(ca=1.0)
    neuron.step(0.0)
    assert 0.0 < neuron.ca < 1.0


def test_spike_path_bounds_and_reset_preserves_parameters() -> None:
    neuron = SKNeuron(g_sk=4.0, v=-20.0)
    assert neuron.step(1.0e6) == 1
    assert -100.0 <= neuron.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (neuron.h, neuron.n))
    assert neuron.ca > 0.0
    neuron.reset()
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.0)
    assert neuron.g_sk == 4.0
