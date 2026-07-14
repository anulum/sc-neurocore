# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBA LIF independent reference-trace contract

"""Independent source-equation reference parity for COBA LIF."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    ReferenceTraceSpec,
    load_reference_trace_spec,
    validate_reference_trace,
)

_TRACE_NAME = "coba_lif_conductance_rk4_doi"
_State = tuple[float, float, float]


def _derivatives(v: float, g_e: float, g_i: float, current: float) -> _State:
    """Evaluate the Brette Benchmark 1 continuous equations."""
    synaptic_current = g_e * v + g_i * (v + 80.0)
    return (
        (-10.0 * (v + 60.0) - synaptic_current + current) / 200.0,
        -g_e / 5.0,
        -g_i / 10.0,
    )


def _rk4(v: float, g_e: float, g_i: float, current: float, dt: float) -> _State:
    """Independently advance the coupled state by classical RK4."""
    k1 = _derivatives(v, g_e, g_i, current)
    k2 = _derivatives(
        v + 0.5 * dt * k1[0],
        g_e + 0.5 * dt * k1[1],
        g_i + 0.5 * dt * k1[2],
        current,
    )
    k3 = _derivatives(
        v + 0.5 * dt * k2[0],
        g_e + 0.5 * dt * k2[1],
        g_i + 0.5 * dt * k2[2],
        current,
    )
    k4 = _derivatives(
        v + dt * k3[0],
        g_e + dt * k3[1],
        g_i + dt * k3[2],
        current,
    )
    scale = dt / 6.0
    return (
        v + scale * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        g_e + scale * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        g_i + scale * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    )


def _rk4_decay(value: float, tau: float, dt: float) -> float:
    """Independently RK4-integrate one exponential conductance decay."""
    k1 = -value / tau
    k2 = -(value + 0.5 * dt * k1) / tau
    k3 = -(value + 0.5 * dt * k2) / tau
    k4 = -(value + dt * k3) / tau
    return value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _independent_features(spec: ReferenceTraceSpec) -> dict[str, float]:
    """Reproduce the complete committed feature map without production helpers."""
    dt = spec.protocol.dt
    current = spec.protocol.inputs["I"]
    delta_ge = spec.protocol.inputs["delta_ge"]
    delta_gi = spec.protocol.inputs["delta_gi"]
    v = -60.0
    g_e = 0.0
    g_i = 0.0
    refractory_time = 0.0
    states: dict[str, list[float]] = {
        "v": [],
        "g_e": [],
        "g_i": [],
        "refractory_time": [],
    }
    spikes: list[int] = []

    for _ in range(spec.protocol.steps):
        g_e += delta_ge
        g_i += delta_gi
        if refractory_time > 0.0:
            v = -60.0
            g_e = _rk4_decay(g_e, 5.0, dt)
            g_i = _rk4_decay(g_i, 10.0, dt)
            refractory_time = (
                0.0 if refractory_time <= dt * (1.0 + 1.0e-12) else refractory_time - dt
            )
            spike = 0
        else:
            candidate_v, g_e, g_i = _rk4(v, g_e, g_i, current, dt)
            spike = int(candidate_v >= -50.0)
            if spike:
                v = -60.0
                refractory_time = 5.0
            else:
                v = candidate_v
        spikes.append(spike)
        states["v"].append(v)
        states["g_e"].append(g_e)
        states["g_i"].append(g_i)
        states["refractory_time"].append(refractory_time)

    features = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for variable, values in states.items():
        features[f"final.{variable}"] = values[-1]
        features[f"min.{variable}"] = min(values)
        features[f"max.{variable}"] = max(values)
        features[f"mean.{variable}"] = math.fsum(values) / len(values)
    return features


def test_trace_features_match_independent_source_equations() -> None:
    """Reproduce every committed feature independently to ``1e-12``."""
    spec = load_reference_trace_spec(_TRACE_NAME)
    expected = _independent_features(spec)

    assert spec.schema_name == "coba_lif"
    assert spec.provenance.kind == "independent_coupled_rk4_reference"
    assert spec.provenance.citation == "doi:10.1007/s10827-007-0038-6"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_schema_runner_matches_committed_reference() -> None:
    """Validate the real four-stage schema runner against the independent trace."""
    report = validate_reference_trace(_TRACE_NAME)

    assert report.passed
    assert report.mismatches == ()
