# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque independent reference-trace contract

"""Independent analytic closed-form reference parity for Lapicque."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec


def _closed_form_features(current: float, dt: float, steps: int) -> dict[str, float]:
    """Derive sampled constant-current RC features without the schema runner."""
    tau = 20.0
    v_rest = 0.0
    resistance = 1.0
    v0 = 0.0
    v_inf = v_rest + resistance * current
    trace = [v_inf + (v0 - v_inf) * math.exp(-((index + 1) * dt) / tau) for index in range(steps)]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": trace[-1],
        "min.v": min(trace),
        "max.v": max(trace),
        "mean.v": sum(trace) / len(trace),
    }


def test_trace_features_match_independent_closed_form() -> None:
    """Reproduce every committed feature independently to ``1e-12``."""
    spec = load_reference_trace_spec("lapicque_constant_current_closed_form")
    expected = _closed_form_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )
    assert spec.schema_name == "lapicque"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1007/s00422-007-0189-6"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1.0e-12)
