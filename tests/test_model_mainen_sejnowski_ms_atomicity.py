# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mainen-Sejnowski atomicity, rate-limit, and legacy contracts

"""Fail-closed, singular-limit, and legacy-configuration contracts."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import MainenSejnowskiNeuron as PublicMainenSejnowski
from sc_neurocore.neurons.models import mainen_sejnowski as module
from sc_neurocore.neurons.models.mainen_sejnowski import MainenSejnowskiNeuron


def _state(neuron: MainenSejnowskiNeuron) -> tuple[float, float, float, float, float]:
    return neuron.vs, neuron.va, neuron.m, neuron.h, neuron.n


def test_public_registry_defaults_and_canonical_anchor() -> None:
    neuron = MainenSejnowskiNeuron()
    assert PublicMainenSejnowski is MainenSejnowskiNeuron
    assert _state(neuron) == (-65.0, -65.0, 0.05, 0.6, 0.3)
    assert neuron.legacy_epsilon_rates is False
    assert neuron.step(10.0) == 0
    assert _state(neuron) == pytest.approx(
        (
            -32.668480035293555,
            200.0,
            0.6007942567015805,
            0.6581322365920295,
            0.398198621809121,
        ),
        abs=1.0e-15,
    )


def test_legacy_epsilon_rates_reproduce_the_historical_python_anchor() -> None:
    neuron = MainenSejnowskiNeuron(legacy_epsilon_rates=True)
    assert neuron.step(10.0) == 0
    assert _state(neuron) == pytest.approx(
        (
            -32.668480035293555,
            200.0,
            0.6007942567015181,
            0.6582308990450022,
            0.3981986218090308,
        ),
        abs=1.0e-15,
    )


def test_linoid_limits_are_exact_and_continuous() -> None:
    assert module._linoid(0.0, 9.0) == 9.0
    assert module._linoid(0.0, 5.0) == 5.0
    for k in (9.0, 5.0):
        assert abs(module._linoid(1e-9, k) - k) < 1e-8
        assert abs(module._linoid(-1e-9, k) - k) < 1e-8
    assert 0.0 <= module._linoid(-4000.0, 9.0) < 1e-100
    assert module._safe_exp(1000.0) == pytest.approx(math.exp(500.0))


@pytest.mark.parametrize("v_singular", (-25.0, -40.0, -65.0, 20.0))
def test_public_step_is_continuous_at_singular_voltages(v_singular: float) -> None:
    exact = MainenSejnowskiNeuron(va=v_singular)
    near = MainenSejnowskiNeuron(va=v_singular + 1e-9)
    exact.step(0.0)
    near.step(0.0)
    deltas = [abs(a - b) for a, b in zip(_state(exact), _state(near), strict=True)]
    assert max(deltas) < 1e-6


def test_legacy_epsilon_rates_return_zero_at_singular_voltage() -> None:
    """The historical defect the canonical linoid path corrects."""

    va = -25.0
    legacy_am = 0.182 * (va + 25.0) / (1.0 - module._safe_exp(-(va + 25.0) / 9.0) + 1e-12)
    assert legacy_am == 0.0
    assert module._linoid(va + 25.0, 9.0) == 9.0


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_atomically(current: float) -> None:
    neuron = MainenSejnowskiNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_nan_drive_no_longer_poisons_the_whole_state() -> None:
    """Regression for NCDBG-047: step(NaN) silently set vs=va=m=h=n=NaN."""

    neuron = MainenSejnowskiNeuron()
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert all(math.isfinite(value) for value in _state(neuron))
    assert neuron.step(1.0) in (0, 1)


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = MainenSejnowskiNeuron()
    neuron.c_s = 0.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="c_s"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = MainenSejnowskiNeuron()
    before = _state(neuron)
    monkeypatch.setattr(module, "_linoid", lambda *_args: math.nan)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("vs", math.nan, ValueError),
        ("vs", -200.1, ValueError),
        ("va", 200.1, ValueError),
        ("m", -0.1, ValueError),
        ("h", 1.1, ValueError),
        ("n", math.inf, ValueError),
        ("kappa", 100.1, ValueError),
        ("g_na", 5000.1, ValueError),
        ("g_k", -0.1, ValueError),
        ("g_l", 5.1, ValueError),
        ("e_na", 29.9, ValueError),
        ("e_k", -69.9, ValueError),
        ("e_l", -49.9, ValueError),
        ("c_s", 0.49, ValueError),
        ("c_a", 0.04, ValueError),
        ("dt", 0.0, ValueError),
        ("v_threshold", 20.1, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(
    field: str, value: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        MainenSejnowskiNeuron(**{field: value})


def test_spike_event_and_reset_preserves_parameters() -> None:
    neuron = MainenSejnowskiNeuron(kappa=20.0, vs=-21.0)
    fired = 0
    for _ in range(50):
        fired += neuron.step(50.0)
        if fired:
            break
    assert fired == 1
    neuron.reset()
    assert _state(neuron) == (-65.0, -65.0, 0.05, 0.6, 0.3)
    assert neuron.kappa == 20.0
