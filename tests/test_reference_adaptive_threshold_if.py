# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold reference-trace contracts

"""Reproduce the committed source-bound trace independently of the hand model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec


def _independent_features(current: float, dt: float, steps: int) -> dict[str, float]:
    v_rest = -65.0
    v_reset = -65.0
    theta_rest = -50.0
    delta_theta = 5.0
    tau_m = 10.0
    tau_theta = 50.0
    decay_v = math.exp(-dt / tau_m)
    decay_theta = math.exp(-dt / tau_theta)
    v = -65.0
    theta = -50.0
    v_trace: list[float] = []
    theta_trace: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        next_v = (v_rest + current) + (v - (v_rest + current)) * decay_v
        next_theta = theta_rest + (theta - theta_rest) * decay_theta
        spike = int(next_v >= next_theta)
        if spike:
            v, theta = v_reset, next_theta + delta_theta
        else:
            v, theta = next_v, next_theta
        v_trace.append(v)
        theta_trace.append(theta)
        spikes.append(spike)

    features = {
        "spike_count": float(sum(spikes)),
        "first_spike_step": float(spikes.index(1) if 1 in spikes else -1),
    }
    for name, values in (("v", v_trace), ("theta", theta_trace)):
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = sum(values) / len(values)
    return features


def test_trace_features_match_independent_exact_relaxation_reference() -> None:
    spec = load_reference_trace_spec("adaptive_threshold_if_tonic_adaptation_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "adaptive_threshold_if"
    assert spec.provenance.kind == "analytic_exact_relaxation_reference"
    assert "doi:10.1162/neco.2008.12-07-680" in spec.provenance.citation
    assert "doi:10.1371/journal.pcbi.1000850" in spec.provenance.citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(
            feature_value,
            rel=0.0,
            abs=1.0e-12,
        )
