# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — T-type source-model contracts

"""Exact source coverage for the WB base + low-voltage-activated IT model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import TTypeCaNeuron as PublicTTypeCaNeuron
from sc_neurocore.neurons.models import ttype_ca_neuron as module
from sc_neurocore.neurons.models.ttype_ca_neuron import TTypeCaNeuron


def _state(neuron: TTypeCaNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.s


def test_public_registry_defaults_and_nominal_anchor() -> None:
    neuron = TTypeCaNeuron()
    assert PublicTTypeCaNeuron is TTypeCaNeuron
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.9)
    assert neuron._sub_steps == 50
    assert neuron.step(5.0) == 0
    assert _state(neuron) == pytest.approx(
        (-63.1681363402518, 0.6480432597760017, 0.23721689617272787, 0.8920254272047233),
        abs=1.0e-15,
    )


def test_rate_singularity_branches() -> None:
    assert module._safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0
    assert module._safe_rate(0.1, 35.0, -34.0, 10.0, 1.0) > 0.0


def test_t_type_gating_voltage_dependence() -> None:
    m_t_rest = 1.0 / (1.0 + math.exp(-(-65.0 + 52.0) / 5.0))
    m_t_depol = 1.0 / (1.0 + math.exp(-(-40.0 + 52.0) / 5.0))
    s_rest = 1.0 / (1.0 + math.exp((-65.0 + 81.0) / 4.0))
    s_hyper = 1.0 / (1.0 + math.exp((-90.0 + 81.0) / 4.0))
    assert m_t_rest < 0.1 < m_t_depol
    assert s_rest < 0.05 < s_hyper


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("v", math.nan, ValueError),
        ("v", -100.1, ValueError),
        ("h", -0.1, ValueError),
        ("n", 1.1, ValueError),
        ("s", 1.1, ValueError),
        ("s", math.inf, ValueError),
        ("g_na", -0.1, ValueError),
        ("g_k", 100.1, ValueError),
        ("g_t", 20.1, ValueError),
        ("g_l", 5.1, ValueError),
        ("e_na", 29.9, ValueError),
        ("e_k", -69.9, ValueError),
        ("e_ca", 59.9, ValueError),
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
        TTypeCaNeuron(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_atomically(current: float) -> None:
    neuron = TTypeCaNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_infinite_drive_no_longer_emits_a_false_spike() -> None:
    """Regression for NCDBG-045: step(+inf) returned 1 and collapsed s to ~1.35e-6."""

    neuron = TTypeCaNeuron()
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.inf)
    assert neuron.s == 0.9
    assert neuron.v == -65.0


def test_corrupted_runtime_configuration_is_rejected_atomically() -> None:
    neuron = TTypeCaNeuron()
    neuron.c_m = 0.0
    before = _state(neuron)
    with pytest.raises(ValueError, match="c_m"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = TTypeCaNeuron()
    before = _state(neuron)
    monkeypatch.setattr(module, "_safe_rate", lambda *_args: math.inf)
    with pytest.raises(ValueError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_spike_collapses_t_type_inactivation() -> None:
    neuron = TTypeCaNeuron()
    fired = 0
    for _ in range(2000):
        fired += neuron.step(5.0)
        if fired:
            break
    assert fired == 1
    assert neuron.s < 0.9


def test_hyperpolarisation_de_inactivates() -> None:
    neuron = TTypeCaNeuron(s=0.2)
    for _ in range(400):
        neuron.step(-3.0)
    assert neuron.s > 0.2


def test_spike_path_bounds_and_reset_preserves_parameters() -> None:
    neuron = TTypeCaNeuron(g_t=0.5, v=-20.0)
    assert neuron.step(1.0e6) == 1
    assert -100.0 <= neuron.v <= 60.0
    assert all(0.0 <= gate <= 1.0 for gate in (neuron.h, neuron.n, neuron.s))
    neuron.reset()
    assert _state(neuron) == (-65.0, 0.6, 0.32, 0.9)
    assert neuron.g_t == 0.5
