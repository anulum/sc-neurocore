# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ih source-model contracts

"""Exact source coverage for the experimental WB+HCN composite."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import IhNeuron as PublicIhNeuron
from sc_neurocore.neurons.models import ih_neuron as module
from sc_neurocore.neurons.models.ih_neuron import IhNeuron


def _state(neuron: IhNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.r


def test_public_registry_defaults_and_nominal_anchor() -> None:
    neuron = IhNeuron()
    assert PublicIhNeuron is IhNeuron
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.1)
    assert neuron._sub_steps == 50
    assert neuron.step(5.0) == 0
    assert _state(neuron) == pytest.approx(
        (-63.049222401527146, 0.6475555997028981, 0.2374056755751358, 0.10020201896411961),
        abs=1.0e-15,
    )


def test_rate_singularity_and_stable_logistic_branches() -> None:
    assert module._safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0
    assert module._safe_rate(0.1, 35.0, -34.0, 10.0, 1.0) > 0.0
    assert module._logistic(1.0) < 0.5
    assert module._logistic(-1.0) > 0.5


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("v", math.nan, ValueError),
        ("v", -100.1, ValueError),
        ("h", -0.1, ValueError),
        ("n", 1.1, ValueError),
        ("r", math.inf, ValueError),
        ("g_na", -0.1, ValueError),
        ("g_k", 100.1, ValueError),
        ("g_h", -0.1, ValueError),
        ("g_l", 5.1, ValueError),
        ("e_na", 29.9, ValueError),
        ("e_k", -69.9, ValueError),
        ("e_h", -50.1, ValueError),
        ("e_l", -39.9, ValueError),
        ("c_m", 0.49, ValueError),
        ("phi", 10.1, ValueError),
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
        IhNeuron(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_atomically(current: float) -> None:
    neuron = IhNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = IhNeuron()
    neuron.c_m = 0.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="c_m"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = IhNeuron()
    before = _state(neuron)
    monkeypatch.setattr(module, "_safe_rate", lambda *_args: math.inf)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_spike_path_bounds_and_reset_preserves_parameters() -> None:
    neuron = IhNeuron(g_h=0.3, v=-20.0)
    assert neuron.step(1.0e6) == 1
    assert -100.0 <= neuron.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (neuron.h, neuron.n, neuron.r))
    neuron.reset()
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.1)
    assert neuron.g_h == 0.3
