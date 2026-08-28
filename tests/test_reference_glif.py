# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF5 and retained recurrence independent receipts

"""Bind both GLIF identities to independent equations and public step APIs."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.neurons.models.sc_four_state_glif import SCFourStateGLIFNeuron
from sc_neurocore.neurons.reference_traces import load_reference_trace_spec


_RECEIPT_ROOT = Path(__file__).parents[1] / "src/sc_neurocore/neurons/reference_receipts"


def _convolution(decay_rate: float, forcing_rate: float, dt: float) -> float:
    difference = decay_rate - forcing_rate
    scale = max(1.0, abs(decay_rate), abs(forcing_rate))
    if abs(difference) <= 1e-12 * scale:
        return dt * math.exp(-decay_rate * dt)
    return (math.exp(-forcing_rate * dt) - math.exp(-decay_rate * dt)) / difference


def _glif5_oracle(drive: tuple[float, ...]) -> tuple[dict[str, Any], dict[str, float]]:
    """Evaluate Teeter equations without importing implementation helpers."""
    v = -70.0
    theta_spike = i_asc1 = i_asc2 = theta_voltage = refractory = 0.0
    events = 0
    first_event = -1
    digest = hashlib.sha256()
    state_traces: list[list[float]] = [[] for _ in range(6)]
    for index, current in enumerate(drive):
        event = 0
        if refractory > 0.0:
            refractory = max(0.0, refractory - 1.0)
        else:
            total_current = current + i_asc1 + i_asc2
            membrane_rate = 1.0 / (1.0 * 10.0)
            equilibrium_offset = 1.0 * total_current
            voltage_offset = v - (-70.0)
            next_offset = equilibrium_offset + (voltage_offset - equilibrium_offset) * math.exp(
                -membrane_rate * 1.0
            )
            v = -70.0 + next_offset
            theta_spike *= math.exp(-0.01 * 1.0)
            i_asc1 *= math.exp(-0.1 * 1.0)
            i_asc2 *= math.exp(-0.005 * 1.0)
            forcing = equilibrium_offset * (1.0 - math.exp(-0.01 * 1.0)) / 0.01 + (
                voltage_offset - equilibrium_offset
            ) * _convolution(0.01, membrane_rate, 1.0)
            theta_voltage = theta_voltage * math.exp(-0.01 * 1.0) + 0.0001 * forcing
            if v > -50.0 + theta_spike + theta_voltage:
                v = -70.0
                theta_spike += 2.0
                i_asc1 += 1.0
                i_asc2 += 0.5
                refractory = 2.0
                event = 1
                events += 1
                if first_event < 0:
                    first_event = index
        state = (v, theta_spike, i_asc1, i_asc2, theta_voltage, refractory)
        for trace, value in zip(state_traces, state, strict=True):
            trace.append(value)
        digest.update(struct.pack("<6dB", *state, event))
    receipt = {
        "events": events,
        "first_event_index": first_event,
        "v_final": v,
        "theta_spike_final": theta_spike,
        "i_asc1_final": i_asc1,
        "i_asc2_final": i_asc2,
        "theta_voltage_final": theta_voltage,
        "refractory_remaining_final": refractory,
        "trace_sha256": digest.hexdigest(),
    }
    return receipt, _features(
        events,
        first_event,
        ("v", "theta_spike", "i_asc1", "i_asc2", "theta_voltage", "refractory_remaining"),
        state_traces,
    )


def _rk4_candidate(state: tuple[float, ...], current: float) -> tuple[float, ...]:
    def derivative(values: tuple[float, ...]) -> tuple[float, ...]:
        v, theta, asc1, asc2 = values
        return (
            (-(v + 70.0) + current + asc1 + asc2) / 10.0,
            (-50.0 - theta + 0.01 * (v + 70.0)) / 100.0,
            -asc1 / 10.0,
            -asc2 / 200.0,
        )

    def add(values: tuple[float, ...], slope: tuple[float, ...], scale: float) -> tuple[float, ...]:
        return tuple(value + scale * delta for value, delta in zip(values, slope, strict=True))

    k1 = derivative(state)
    k2 = derivative(add(state, k1, 0.5))
    k3 = derivative(add(state, k2, 0.5))
    k4 = derivative(add(state, k3, 1.0))
    return tuple(
        value + (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
        for value, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4, strict=True)
    )


def _sc_oracle(drive: tuple[float, ...]) -> tuple[dict[str, Any], dict[str, float]]:
    state = (-70.0, -50.0, 0.0, 0.0)
    events = 0
    first_event = -1
    digest = hashlib.sha256()
    state_traces: list[list[float]] = [[] for _ in range(4)]
    for index, current in enumerate(drive):
        v, theta, i_asc1, i_asc2 = _rk4_candidate(state, current)
        event = int(v >= theta)
        if event:
            v = -70.0
            theta += 2.0
            i_asc1 += 1.0
            i_asc2 += 0.5
            events += 1
            if first_event < 0:
                first_event = index
        state = (v, theta, i_asc1, i_asc2)
        for trace, value in zip(state_traces, state, strict=True):
            trace.append(value)
        digest.update(struct.pack("<4dB", *state, event))
    receipt = {
        "events": events,
        "first_event_index": first_event,
        "v_final": state[0],
        "theta_final": state[1],
        "i_asc1_final": state[2],
        "i_asc2_final": state[3],
        "trace_sha256": digest.hexdigest(),
    }
    return receipt, _features(events, first_event, ("v", "theta", "i_asc1", "i_asc2"), state_traces)


def _features(
    events: int,
    first_event: int,
    names: tuple[str, ...],
    traces: list[list[float]],
) -> dict[str, float]:
    features = {
        "spike_count": float(events),
        "first_spike_step": float(first_event + 1 if first_event >= 0 else -1),
    }
    for name, trace in zip(names, traces, strict=True):
        features[f"final.{name}"] = trace[-1]
        features[f"min.{name}"] = min(trace)
        features[f"max.{name}"] = max(trace)
        features[f"mean.{name}"] = math.fsum(trace) / len(trace)
    return features


def _public_receipt(
    model: GLIFNeuron | SCFourStateGLIFNeuron, drive: tuple[float, ...]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    events = 0
    first_event = -1
    for index, current in enumerate(drive):
        event = model.step(current)
        if event and first_event < 0:
            first_event = index
        events += event
        if isinstance(model, GLIFNeuron):
            canonical_state = (
                model.v,
                model.theta_spike,
                model.i_asc1,
                model.i_asc2,
                model.theta_voltage,
                model.refractory_remaining,
            )
            digest.update(struct.pack("<6dB", *canonical_state, event))
        else:
            retained_state = (model.v, model.theta, model.i_asc1, model.i_asc2)
            digest.update(struct.pack("<4dB", *retained_state, event))
    if isinstance(model, GLIFNeuron):
        return {
            "events": events,
            "first_event_index": first_event,
            "v_final": model.v,
            "theta_spike_final": model.theta_spike,
            "i_asc1_final": model.i_asc1,
            "i_asc2_final": model.i_asc2,
            "theta_voltage_final": model.theta_voltage,
            "refractory_remaining_final": model.refractory_remaining,
            "trace_sha256": digest.hexdigest(),
        }
    return {
        "events": events,
        "first_event_index": first_event,
        "v_final": model.v,
        "theta_final": model.theta,
        "i_asc1_final": model.i_asc1,
        "i_asc2_final": model.i_asc2,
        "trace_sha256": digest.hexdigest(),
    }


@pytest.mark.parametrize(
    ("receipt_name", "model_type", "oracle"),
    [
        ("glif5_teeter_2018.json", GLIFNeuron, _glif5_oracle),
        ("sc_four_state_glif_project.json", SCFourStateGLIFNeuron, _sc_oracle),
    ],
)
def test_complete_receipt_matches_independent_equations_and_public_model(
    receipt_name: str,
    model_type: type[GLIFNeuron] | type[SCFourStateGLIFNeuron],
    oracle: Any,
) -> None:
    receipt = json.loads((_RECEIPT_ROOT / receipt_name).read_text(encoding="utf-8"))
    pattern = tuple(float(value) for value in receipt["drive"]["pattern"])
    drive = pattern * int(receipt["drive"]["repeats"])
    expected, _ = oracle(drive)
    assert receipt["oracle"] == expected
    assert receipt["oracle"] == _public_receipt(model_type(), drive)


@pytest.mark.parametrize(
    ("trace_name", "schema_name", "kind", "citation", "oracle"),
    [
        (
            "glif_constant_current_threshold_adaptation",
            "glif",
            "independent_exact_flow_reference",
            "doi:10.1038/s41467-017-02717-4",
            _glif5_oracle,
        ),
        (
            "sc_four_state_glif_constant_current_adaptation",
            "sc_four_state_glif",
            "project_rk4_reference",
            "project:sc-neurocore-four-state-glif",
            _sc_oracle,
        ),
    ],
)
def test_trace_features_match_independent_equations(
    trace_name: str,
    schema_name: str,
    kind: str,
    citation: str,
    oracle: Any,
) -> None:
    spec = load_reference_trace_spec(trace_name)
    _, expected = oracle((spec.protocol.inputs["I"],) * spec.protocol.steps)
    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
