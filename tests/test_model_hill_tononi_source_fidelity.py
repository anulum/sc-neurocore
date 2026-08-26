# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Independent Hill-Tononi 2005 recurrence receipt

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import tomllib
from typing import cast

import pytest

from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

_State = tuple[float, float, float, float, float, float]
_RECEIPT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_receipts/hill_tononi_2005.json"
)
_SCHEMA = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas/hill_tononi"


def _reference_derivatives(state: _State, current: float, spike_active: bool) -> _State:
    v, theta, d_k, m_h, m_t, h_t = state
    m_na_p = 1.0 / (1.0 + math.exp(-(v + 55.7) / 7.7))
    d_inf = 1250.0 * 0.025 / (1.0 + math.exp(-(v + 10.0) / 5.0)) + 0.001
    d_activation = 1.0 / (1.0 + (0.25 / max(d_k, 1e-15)) ** 3.5)
    m_h_inf = 1.0 / (1.0 + math.exp((v + 75.0) / 5.5))
    tau_m_h = 1.0 / (math.exp(-14.59 - 0.086 * v) + math.exp(-1.87 + 0.0701 * v))
    m_t_inf = 1.0 / (1.0 + math.exp(-(v + 59.0) / 6.2))
    tau_m_t = 0.22 / (math.exp(-(v + 132.0) / 16.7) + math.exp((v + 16.8) / 18.2)) + 0.13
    h_t_inf = 1.0 / (1.0 + math.exp((v + 83.0) / 4.0))
    tau_h_t = 8.2 + (56.6 + 0.27 * math.exp((v + 115.2) / 5.0)) / (1.0 + math.exp((v + 86.0) / 3.2))

    i_na_l = -0.2 * (v - 30.0)
    i_k_l = -(v + 90.0)
    i_na_p = -0.5 * m_na_p**3 * (v - 30.0)
    i_dk = -0.5 * d_activation * (v + 90.0)
    i_spike = -(v + 90.0) / 1.75 if spike_active else 0.0
    return (
        (i_na_l + i_k_l + i_na_p + i_dk + current) / 16.0 + i_spike,
        -(theta + 51.0) / 2.0,
        (d_inf - d_k) / 1250.0,
        (m_h_inf - m_h) / tau_m_h,
        (m_t_inf - m_t) / tau_m_t,
        (h_t_inf - h_t) / tau_h_t,
    )


def _reference_rk4(state: _State, current: float, spike_active: bool) -> _State:
    dt = 0.25
    k1 = _reference_derivatives(state, current, spike_active)
    s2 = cast(_State, tuple(value + 0.5 * dt * slope for value, slope in zip(state, k1)))
    k2 = _reference_derivatives(s2, current, spike_active)
    s3 = cast(_State, tuple(value + 0.5 * dt * slope for value, slope in zip(state, k2)))
    k3 = _reference_derivatives(s3, current, spike_active)
    s4 = cast(_State, tuple(value + dt * slope for value, slope in zip(state, k3)))
    k4 = _reference_derivatives(s4, current, spike_active)
    return cast(
        _State,
        tuple(
            value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(state, k1, k2, k3, k4)
        ),
    )


def _reference_step(state: _State, timer: float, current: float) -> tuple[_State, float, int]:
    refractory = timer > 0.0
    candidate = _reference_rk4(state, current, refractory)
    next_timer = max(0.0, timer - 0.25)
    spike = int(not refractory and candidate[0] >= candidate[1])
    if spike:
        candidate = (30.0, 30.0, *candidate[2:])
        next_timer = 2.0
    return candidate, next_timer, spike


def test_source_cortical_wake_defaults() -> None:
    neuron = HillTononiNeuron()
    assert (neuron.v, neuron.theta, neuron.d_k, neuron.dt) == (-70.0, -51.0, 0.001, 0.25)
    assert (neuron.g_na_l, neuron.g_k_l) == (0.2, 1.0)
    assert (neuron.g_na_p, neuron.g_dk) == (0.5, 0.5)
    assert (neuron.g_h, neuron.g_t) == (0.0, 0.0)
    assert not hasattr(neuron, "na_i")


def test_independent_rk4_receipt_matches_public_step() -> None:
    neuron = HillTononiNeuron()
    state = (neuron.v, neuron.theta, neuron.d_k, neuron.m_h, neuron.m_t, neuron.h_t)
    expected = _reference_rk4(state, current=12.0, spike_active=False)
    assert neuron.step(12.0) == 0
    observed = (neuron.v, neuron.theta, neuron.d_k, neuron.m_h, neuron.m_t, neuron.h_t)
    assert observed == pytest.approx(expected, rel=0.0, abs=2e-14)


def test_dynamic_threshold_and_post_spike_pulse() -> None:
    neuron = HillTononiNeuron(v=-50.0, theta=-51.0)
    assert neuron.step(0.0) == 1
    assert (neuron.v, neuron.theta, neuron.spike_timer) == (30.0, 30.0, 2.0)
    assert neuron.step(0.0) == 0
    assert neuron.spike_timer == 1.75
    assert neuron.v < 30.0


def test_source_gating_anchors() -> None:
    neuron = HillTononiNeuron()
    assert neuron.m_h_inf(-75.0) == 0.5
    assert neuron.m_t_inf(-59.0) == 0.5
    assert neuron.h_t_inf(-83.0) == 0.5
    assert neuron.d_k_inf(-10.0) == pytest.approx(15.626, rel=0.0, abs=1e-15)


def test_optional_thalamic_currents_evolve_finitely() -> None:
    neuron = HillTononiNeuron(g_h=1.0, g_t=1.0)
    for _ in range(200):
        neuron.step(0.0)
    assert all(math.isfinite(value) for value in (neuron.v, neuron.m_h, neuron.m_t, neuron.h_t))


def test_invalid_runtime_configuration_is_atomic() -> None:
    neuron = HillTononiNeuron()
    before = (neuron.v, neuron.theta, neuron.d_k, neuron.m_h, neuron.m_t, neuron.h_t)
    neuron.dt = math.nan
    with pytest.raises(ValueError, match="dt"):
        neuron.step(0.0)
    after = (neuron.v, neuron.theta, neuron.d_k, neuron.m_h, neuron.m_t, neuron.h_t)
    assert after == before


def test_reset_restores_source_state() -> None:
    neuron = HillTononiNeuron()
    for _ in range(200):
        neuron.step(20.0)
    neuron.reset()
    assert (neuron.v, neuron.theta, neuron.d_k, neuron.spike_timer) == (
        -70.0,
        -51.0,
        0.001,
        0.0,
    )
    assert neuron.m_h == neuron.m_h_inf(-70.0)
    assert neuron.m_t == neuron.m_t_inf(-70.0)
    assert neuron.h_t == neuron.h_t_inf(-70.0)


def test_independent_mixed_drive_receipt() -> None:
    receipt = json.loads(_RECEIPT.read_text(encoding="utf-8"))
    pattern = tuple(float(value) for value in receipt["drive"]["pattern"])
    repeats = int(receipt["drive"]["repeats"])
    state: _State = (
        -70.0,
        -51.0,
        0.001,
        0.2871859013825026,
        0.1450215950687922,
        0.03732688734412946,
    )
    timer = 0.0
    events = 0
    digest = hashlib.sha256()
    neuron = HillTononiNeuron()
    for current in pattern * repeats:
        state, timer, event = _reference_step(state, timer, current)
        observed_event = neuron.step(current)
        observed = (neuron.v, neuron.theta, neuron.d_k, neuron.m_h, neuron.m_t, neuron.h_t)
        assert observed == pytest.approx(state, rel=0.0, abs=2e-13)
        assert (neuron.spike_timer, observed_event) == (timer, event)
        events += event
        digest.update(struct.pack("<6dq", *state, event))
    oracle = receipt["oracle"]
    assert events == oracle["events"]
    assert list(state) == pytest.approx(oracle["final_state"], rel=0.0, abs=2e-13)
    assert timer == oracle["spike_timer_final"]
    assert digest.hexdigest() == oracle["trace_sha256"]


def test_paired_schemas_preserve_source_identity() -> None:
    with _SCHEMA.with_suffix(".toml").open("rb") as handle:
        toml_payload = tomllib.load(handle)
    json_payload = json.loads(_SCHEMA.with_suffix(".json").read_text(encoding="utf-8"))
    assert toml_payload == json_payload
    assert tuple(toml_payload["state"]) == (
        "v",
        "theta",
        "d_k",
        "m_h",
        "m_t",
        "h_t",
        "spike_timer",
    )
    assert toml_payload["metadata"]["doi"] == "10.1152/jn.00915.2004"
    assert "na_i" not in toml_payload["state"]
