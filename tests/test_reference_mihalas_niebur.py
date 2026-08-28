# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur and retained-SC independent receipts

"""Bind both Model 14 identities to independent equations and public APIs."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)
from sc_neurocore.neurons.reference_traces import ReferenceTraceSpec, load_reference_trace_spec
from tests.cosim_reference_mihalas_niebur import (
    _mihalas_niebur_driven_rk4_features,
    _sc_scaled_reset_driven_rk4_features,
)

_ROOT = Path(__file__).parents[1]
_RECEIPT_ROOT = _ROOT / "src/sc_neurocore/neurons/reference_receipts"
State = tuple[float, float, float, float]


def _rk4(state: State, dt: float, derivative: Callable[[State], State]) -> State:
    def add(values: State, slope: State, scale: float) -> State:
        return cast(
            State,
            tuple(value + scale * delta for value, delta in zip(values, slope, strict=True)),
        )

    k1 = derivative(state)
    k2 = derivative(add(state, k1, 0.5 * dt))
    k3 = derivative(add(state, k2, 0.5 * dt))
    k4 = derivative(add(state, k3, dt))
    return cast(
        State,
        tuple(
            value + dt * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0
            for value, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4, strict=True)
        ),
    )


def _independent_receipt(*, source_model: bool, drive: tuple[float, ...]) -> dict[str, Any]:
    state: State = (-0.07, -0.05, 0.0, 0.0) if source_model else (0.0, 1.0, 0.0, 0.0)
    dt = 0.1 if source_model else 1.0
    digest = hashlib.sha256()
    events = 0
    first_event = -1
    for index, current in enumerate(drive):
        if source_model:

            def derivative(values: State) -> State:
                v, theta, i1, i2 = values
                return (
                    current + i1 + i2 - 0.05 * (v + 0.07),
                    0.005 * (v + 0.07) - 0.01 * (theta + 0.05),
                    -0.2 * i1,
                    -0.02 * i2,
                )

        else:

            def derivative(values: State) -> State:
                v, theta, i1, i2 = values
                return (
                    (-v + i1 + i2 + current) / 10.0,
                    (1.0 - theta + 0.1 * v) / 40.0,
                    -i1 / 15.0,
                    -i2 / 80.0,
                )

        v, theta, i1, i2 = _rk4(state, dt, derivative)
        event = int(v >= theta)
        if event:
            if source_model:
                v, theta, i1, i2 = -0.07, max(-0.06, theta), 0.01, i2 - 0.0006
            else:
                v, theta, i1, i2 = 0.1 * v, max(1.3, theta), i1 + 0.2, i2 - 0.15
            events += 1
            if first_event < 0:
                first_event = index
        state = (v, theta, i1, i2)
        digest.update(struct.pack("<4dB", *state, event))
    return {
        "events": events,
        "first_event_index": first_event,
        "v_final": state[0],
        "theta_final": state[1],
        "i1_final": state[2],
        "i2_final": state[3],
        "trace_sha256": digest.hexdigest(),
    }


def _public_receipt(
    model: MihalasNieburNeuron | SCScaledResetAdaptiveIFNeuron,
    drive: tuple[float, ...],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    events = 0
    first_event = -1
    for index, current in enumerate(drive):
        event = model.step(current)
        events += event
        if event and first_event < 0:
            first_event = index
        digest.update(struct.pack("<4dB", model.v, model.theta, model.i1, model.i2, event))
    return {
        "events": events,
        "first_event_index": first_event,
        "v_final": model.v,
        "theta_final": model.theta,
        "i1_final": model.i1,
        "i2_final": model.i2,
        "trace_sha256": digest.hexdigest(),
    }


@pytest.mark.parametrize(
    ("receipt_name", "source_model", "model"),
    [
        (
            "mihalas_niebur_2009.json",
            True,
            MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006),
        ),
        (
            "sc_scaled_reset_adaptive_if_project.json",
            False,
            SCScaledResetAdaptiveIFNeuron(
                theta_reset=1.3,
                tau_theta=40.0,
                tau_1=15.0,
                tau_2=80.0,
                a=0.1,
                b=0.1,
                r1=0.2,
                r2=-0.15,
            ),
        ),
    ],
)
def test_complete_receipt_matches_independent_equations_and_public_model(
    receipt_name: str,
    source_model: bool,
    model: MihalasNieburNeuron | SCScaledResetAdaptiveIFNeuron,
) -> None:
    receipt = json.loads((_RECEIPT_ROOT / receipt_name).read_text(encoding="utf-8"))
    pattern = tuple(float(value) for value in receipt["drive"]["pattern"])
    drive = pattern * int(receipt["drive"]["repeats"])
    assert receipt["oracle"] == _independent_receipt(source_model=source_model, drive=drive)
    assert receipt["oracle"] == _public_receipt(model, drive)


_PARITY_CASES: list[tuple[str, str, str, str, Callable[[ReferenceTraceSpec], dict[str, float]]]] = [
    (
        "mihalas_niebur_driven_spiking_doi",
        "mihalas_niebur",
        "independent_rk4_reference",
        "doi:10.1162/neco.2008.12-07-680",
        lambda spec: _mihalas_niebur_driven_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
    (
        "sc_scaled_reset_adaptive_if_driven_project",
        "sc_scaled_reset_adaptive_if",
        "project_rk4_reference",
        "project:sc-neurocore-scaled-reset-adaptive-if",
        lambda spec: _sc_scaled_reset_driven_rk4_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
]


@pytest.mark.parametrize(
    ("trace_name", "schema_name", "kind", "citation", "reference"),
    _PARITY_CASES,
    ids=[case[1] for case in _PARITY_CASES],
)
def test_trace_features_match_independent_reference(
    trace_name: str,
    schema_name: str,
    kind: str,
    citation: str,
    reference: Callable[[ReferenceTraceSpec], dict[str, float]],
) -> None:
    spec = load_reference_trace_spec(trace_name)
    expected = reference(spec)
    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    if schema_name == "mihalas_niebur":
        assert spec.protocol.parameter_overrides == {
            "current_jump_1": 0.01,
            "current_jump_2": -0.0006,
        }
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
