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


def _closed_form_features(source_voltage: float, dt: float, steps: int) -> dict[str, float]:
    """Derive source polarization/latch features without the schema runner."""
    v_inf = source_voltage / 11.0
    trace = [v_inf * (1.0 - math.exp(-((index + 1) * dt))) for index in range(steps)]
    event_index = next(index for index, voltage in enumerate(trace) if voltage >= 1.0)
    excited = [float(index >= event_index) for index in range(steps)]
    return {
        "spike_count": 1.0,
        "first_spike_step": float(event_index + 1),
        "final.v": trace[-1],
        "min.v": min(trace),
        "max.v": max(trace),
        "mean.v": sum(trace) / len(trace),
        "final.excited": excited[-1],
        "min.excited": min(excited),
        "max.excited": max(excited),
        "mean.excited": sum(excited) / len(excited),
    }


def test_trace_features_match_independent_closed_form() -> None:
    """Reproduce every committed feature independently to ``1e-12``."""
    spec = load_reference_trace_spec("lapicque_1907_constant_voltage_closed_form")
    expected = _closed_form_features(
        source_voltage=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )
    assert spec.schema_name == "lapicque"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1007/s00422-007-0189-6"
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1.0e-12)


def test_sc_compatibility_trace_is_retained_without_source_attribution() -> None:
    spec = load_reference_trace_spec("sc_lapicque_lif_constant_current_closed_form")
    assert spec.schema_name == "sc_lapicque_lif"
    assert spec.provenance.source.startswith("SC-NeuroCore retained project recurrence")
    assert spec.provenance.citation == "SC-NeuroCore count-neutral compatibility profile"
