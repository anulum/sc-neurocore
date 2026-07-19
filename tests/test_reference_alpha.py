# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse reference-trace contracts

"""Reproduce the committed source-bound trace independently of the hand model."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec


def _drive_contribution(
    current_delta: float,
    rise_delta: float,
    tau_drive: float,
    tau_v: float,
    dt: float,
) -> float:
    rate_v = 1.0 / tau_v
    rate_drive = 1.0 / tau_drive
    decay_v = math.exp(-dt / tau_v)
    decay_drive = math.exp(-dt / tau_drive)
    rate_delta = rate_v - rate_drive
    first_order = current_delta * (decay_drive - decay_v) / rate_delta
    second_order = (
        rise_delta
        / tau_drive
        * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
        / (rate_delta * rate_delta)
    )
    return rate_v * (first_order + second_order)


def _independent_features(
    i_exc_drive: float, i_inh_drive: float, dt: float, steps: int
) -> dict[str, float]:
    v_rest = 0.0
    v_threshold = 1.0
    tau_v = 20.0
    tau_exc = 5.0
    tau_inh = 10.0
    v = 0.0
    a_exc = 0.0
    i_exc = 0.0
    a_inh = 0.0
    i_inh = 0.0
    traces: dict[str, list[float]] = {
        "v": [],
        "a_exc": [],
        "i_exc": [],
        "a_inh": [],
        "i_inh": [],
    }
    spikes: list[int] = []
    for _ in range(steps):
        exc_steady = tau_exc * i_exc_drive
        inh_steady = tau_inh * i_inh_drive
        a_exc_next = exc_steady + (a_exc - exc_steady) * math.exp(-dt / tau_exc)
        i_exc_next = exc_steady + math.exp(-dt / tau_exc) * (
            (i_exc - exc_steady) + (a_exc - exc_steady) * dt / tau_exc
        )
        a_inh_next = inh_steady + (a_inh - inh_steady) * math.exp(-dt / tau_inh)
        i_inh_next = inh_steady + math.exp(-dt / tau_inh) * (
            (i_inh - inh_steady) + (a_inh - inh_steady) * dt / tau_inh
        )
        v_steady = v_rest + exc_steady - inh_steady
        v_next = (
            v_steady
            + (v - v_steady) * math.exp(-dt / tau_v)
            + _drive_contribution(i_exc - exc_steady, a_exc - exc_steady, tau_exc, tau_v, dt)
            - _drive_contribution(i_inh - inh_steady, a_inh - inh_steady, tau_inh, tau_v, dt)
        )
        spike = int(v_next >= v_threshold)
        a_exc, i_exc, a_inh, i_inh = a_exc_next, i_exc_next, a_inh_next, i_inh_next
        v = v_rest if spike else v_next
        for key, value in (
            ("v", v),
            ("a_exc", a_exc),
            ("i_exc", i_exc),
            ("a_inh", a_inh),
            ("i_inh", i_inh),
        ):
            traces[key].append(value)
        spikes.append(spike)

    features = {
        "spike_count": float(sum(spikes)),
        "first_spike_step": float(spikes.index(1) if 1 in spikes else -1),
    }
    for name, values in traces.items():
        features[f"final.{name}"] = values[-1]
        features[f"min.{name}"] = min(values)
        features[f"max.{name}"] = max(values)
        features[f"mean.{name}"] = sum(values) / len(values)
    return features


def test_trace_features_match_independent_exact_flow_reference() -> None:
    spec = load_reference_trace_spec("alpha_dual_synapse_doi")
    expected = _independent_features(
        i_exc_drive=spec.protocol.inputs["I_exc"],
        i_inh_drive=spec.protocol.inputs["I_inh"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "alpha"
    assert spec.provenance.kind == "analytic_exact_flow_reference"
    assert "doi:10.1017/CBO9780511815706" in spec.provenance.citation
    assert "Rall 1967" in spec.provenance.citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(
            feature_value,
            rel=0.0,
            abs=1.0e-12,
        )
