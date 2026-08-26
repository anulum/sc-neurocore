# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Butera-Rinzel-Smith 1999 Model 1 source fidelity

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron
from sc_neurocore.neurons.models.sc_unit_capacitance_respiratory import (
    SCUnitCapacitanceRespiratoryNeuron,
)

State = tuple[float, float, float]
_ROOT = Path(__file__).resolve().parents[1]


def _source_derivatives(neuron: ButeraRespiratoryNeuron, state: State, i_app: float) -> State:
    """Evaluate the paper equations independently of production helpers."""
    v, n, h_nap = state
    m_na = 1.0 / (1.0 + math.exp(-(v + 34.0) / 5.0))
    m_nap = 1.0 / (1.0 + math.exp(-(v + 40.0) / 6.0))
    h_inf = 1.0 / (1.0 + math.exp((v + 48.0) / 6.0))
    n_inf = 1.0 / (1.0 + math.exp(-(v + 29.0) / 4.0))
    tau_n = 10.0 / math.cosh((v + 29.0) / 8.0)
    tau_h = neuron.tau_h / math.cosh((v + 48.0) / 12.0)
    i_na = neuron.g_na * m_na**3 * (1.0 - n) * (v - neuron.e_na)
    i_nap = neuron.g_nap * m_nap * h_nap * (v - neuron.e_na)
    i_k = neuron.g_k * n**4 * (v - neuron.e_k)
    i_l = neuron.g_l * (v - neuron.e_l)
    i_tonic = neuron.g_tonic * (v - neuron.e_syn)
    return (
        (-i_na - i_nap - i_k - i_l - i_tonic + i_app) / neuron.capacitance,
        (n_inf - n) / tau_n,
        (h_inf - h_nap) / tau_h,
    )


def _source_rk4(neuron: ButeraRespiratoryNeuron, i_app: float) -> State:
    state = (neuron.v, neuron.n, neuron.h_nap)
    dt = neuron.dt
    k1 = _source_derivatives(neuron, state, i_app)
    k2_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
    k2 = _source_derivatives(neuron, k2_state, i_app)  # type: ignore[arg-type]
    k3_state = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
    k3 = _source_derivatives(neuron, k3_state, i_app)  # type: ignore[arg-type]
    k4_state = tuple(s + dt * k for s, k in zip(state, k3))
    k4 = _source_derivatives(neuron, k4_state, i_app)  # type: ignore[arg-type]
    return tuple(
        s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0 for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )  # type: ignore[return-value]


def _sc_project_rk4(neuron: SCUnitCapacitanceRespiratoryNeuron, current: float) -> State:
    """Evaluate the frozen SC project recurrence independently."""

    def project_derivatives(state: State) -> State:
        v, n, h_nap = state
        bounded = (
            max(-200.0, min(100.0, v)),
            max(0.0, min(1.0, n)),
            max(0.0, min(1.0, h_nap)),
        )
        return _source_derivatives(neuron, bounded, current)

    state = (neuron.v, neuron.n, neuron.h_nap)
    dt = neuron.dt
    k1 = project_derivatives(state)
    k2 = project_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k1)))  # type: ignore[arg-type]
    k3 = project_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k2)))  # type: ignore[arg-type]
    k4 = project_derivatives(tuple(s + dt * k for s, k in zip(state, k3)))  # type: ignore[arg-type]
    raw = tuple(
        s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0 for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )
    return (
        max(-200.0, min(100.0, raw[0])),
        max(0.0, min(1.0, raw[1])),
        max(0.0, min(1.0, raw[2])),
    )


def test_source_defaults_identify_model_one() -> None:
    neuron = ButeraRespiratoryNeuron()
    assert neuron.capacitance == 21.0
    assert neuron.g_na == 28.0
    assert neuron.g_nap == 2.8
    assert neuron.g_k == 11.2
    assert neuron.g_l == 2.8
    assert neuron.g_tonic == 0.0
    assert neuron.e_syn == 0.0
    assert neuron.dt == 0.1


def test_source_step_matches_independent_paper_equations() -> None:
    neuron = ButeraRespiratoryNeuron(
        v=-52.0,
        n=0.07,
        h_nap=0.61,
        g_tonic=0.35,
        e_syn=-10.0,
        dt=0.025,
    )
    expected = _source_rk4(neuron, i_app=12.5)
    event = neuron.step(12.5)
    assert event in (0, 1)
    assert (neuron.v, neuron.n, neuron.h_nap) == pytest.approx(expected, abs=2e-14)


def test_capacitance_divides_only_the_voltage_equation() -> None:
    source = ButeraRespiratoryNeuron(v=-50.0, n=0.01, h_nap=0.5)
    legacy = SCUnitCapacitanceRespiratoryNeuron(v=-50.0, n=0.01, h_nap=0.5)
    source_dv, source_dn, source_dh = _source_derivatives(source, (-50.0, 0.01, 0.5), 20.0)
    legacy_dv, legacy_dn, legacy_dh = _source_derivatives(legacy, (-50.0, 0.01, 0.5), 20.0)
    assert legacy_dv == pytest.approx(21.0 * source_dv)
    assert legacy_dn == source_dn
    assert legacy_dh == source_dh


def test_source_spike_is_observational_without_reset() -> None:
    neuron = ButeraRespiratoryNeuron(v=-20.01, n=0.2, h_nap=0.5, v_threshold=-20.0, dt=0.001)
    event = neuron.step(200.0)
    assert event == 1
    assert neuron.v != neuron.e_l
    assert neuron.n != 0.01
    assert neuron.h_nap != 0.5


@pytest.mark.parametrize(("field", "value"), [("capacitance", 0.0), ("g_tonic", -0.1)])
def test_source_rejects_invalid_new_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        ButeraRespiratoryNeuron(**{field: value})


def test_source_failure_is_atomic() -> None:
    neuron = ButeraRespiratoryNeuron(v=-49.0, n=0.04, h_nap=0.55)
    before = (neuron.v, neuron.n, neuron.h_nap)
    with pytest.raises((ValueError, FloatingPointError)):
        neuron.step(float("nan"))
    assert (neuron.v, neuron.n, neuron.h_nap) == before


def test_source_enrolled_event_counts() -> None:
    counts = []
    for current in (0.0, 20.0, 50.0):
        neuron = ButeraRespiratoryNeuron()
        counts.append(sum(neuron.step(current) for _ in range(20_000)))
    assert counts == [0, 11, 173]


def test_independent_mixed_drive_matches_frozen_source_receipt() -> None:
    receipt = json.loads(
        (
            _ROOT
            / "src/sc_neurocore/neurons/reference_receipts/butera_rinzel_smith_1999_model1.json"
        ).read_text(encoding="utf-8")
    )
    neuron = ButeraRespiratoryNeuron()
    digest = hashlib.sha256()
    event_indices: list[int] = []
    pattern = [0.0] * 64 + [50.0] * 128 + [-10.0] * 64
    for index, current in enumerate(pattern * 4):
        previous_v = neuron.v
        neuron.v, neuron.n, neuron.h_nap = _source_rk4(neuron, current)
        event = int(neuron.v >= neuron.v_threshold and previous_v < neuron.v_threshold)
        if event:
            event_indices.append(index)
        digest.update(struct.pack("<dddq", neuron.v, neuron.n, neuron.h_nap, event))
    oracle = receipt["oracle"]
    assert oracle["steps"] == 1024
    assert event_indices == oracle["event_indices"]
    assert len(event_indices) == oracle["events"]
    assert [neuron.v, neuron.n, neuron.h_nap] == pytest.approx(oracle["final_state"], abs=2e-14)
    assert digest.hexdigest() == oracle["trace_sha256"]


def test_independent_sc_project_trace_matches_frozen_receipt() -> None:
    receipt = json.loads(
        (
            _ROOT
            / "src/sc_neurocore/neurons/reference_receipts/sc_unit_capacitance_respiratory_v1.json"
        ).read_text(encoding="utf-8")
    )
    neuron = SCUnitCapacitanceRespiratoryNeuron()
    digest = hashlib.sha256()
    event_indices: list[int] = []
    pattern = [0.0] * 64 + [20.0] * 128 + [-5.0] * 64
    for index, current in enumerate(pattern * 4):
        previous_v = neuron.v
        neuron.v, neuron.n, neuron.h_nap = _sc_project_rk4(neuron, current)
        event = int(neuron.v >= neuron.v_threshold and previous_v < neuron.v_threshold)
        if event:
            event_indices.append(index)
        digest.update(struct.pack("<dddq", neuron.v, neuron.n, neuron.h_nap, event))
    oracle = receipt["oracle"]
    assert event_indices == oracle["event_indices"]
    assert [neuron.v, neuron.n, neuron.h_nap] == pytest.approx(oracle["final_state"], abs=0.0)
    assert digest.hexdigest() == oracle["trace_sha256"]
