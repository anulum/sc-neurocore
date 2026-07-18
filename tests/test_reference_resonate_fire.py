# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire reference-trace contracts

"""Reproduce the committed source-bound trace independently of the hand model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec


def _independent_features(current: float, dt: float, steps: int) -> dict[str, float]:
    b = -1.0
    omega = 10.0
    threshold = 1.0
    denominator = b * b + omega * omega
    x_ss = -b * current / denominator
    y_ss = omega * current / denominator
    decay = math.exp(b * dt)
    angle = omega * dt
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x = 0.0
    y = 0.0
    x_trace: list[float] = []
    y_trace: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        next_x = x_ss + decay * ((x - x_ss) * cos_angle - (y - y_ss) * sin_angle)
        next_y = y_ss + decay * ((x - x_ss) * sin_angle + (y - y_ss) * cos_angle)
        spike = int(y < threshold <= next_y)
        if spike:
            x, y = 0.0, threshold
        else:
            x, y = next_x, next_y
        x_trace.append(x)
        y_trace.append(y)
        spikes.append(spike)

    features = {
        "spike_count": float(sum(spikes)),
        "first_spike_step": float(spikes.index(1) if 1 in spikes else -1),
    }
    for name, values in (("x", x_trace), ("y", y_trace)):
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = sum(values) / len(values)
    return features


def test_trace_features_match_independent_exact_flow_reference() -> None:
    spec = load_reference_trace_spec("resonate_fire_subthreshold_resonance_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "resonate_fire"
    assert spec.provenance.kind == "analytic_exact_linear_flow_reference"
    assert spec.provenance.citation == "doi:10.1016/S0893-6080(01)00078-8"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(
            feature_value,
            rel=0.0,
            abs=1.0e-12,
        )
