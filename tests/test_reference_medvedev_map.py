# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Medvedev 2005 first-return reference trace

"""Independent feature derivation for the committed Medvedev DOI trace."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.reference_traces import (
    load_reference_trace_spec,
    validate_reference_trace_spec,
)


def _independent_features(*, current: float, steps: int) -> dict[str, float]:
    """Re-derive the disclosed Section-4 calibration without model code."""
    beta_0 = 0.0015
    beta_hc = 0.00203
    beta_sn = 0.002009000318382601
    delta = 0.01
    decay_t0 = 0.9903563355786734
    alpha_t0 = 0.0096904656865853
    f_0 = 1.4713541429802286
    f_1 = 0.1820152787145665
    homoclinic_exponent = 0.02149298991339221
    d = 2271.1927977404063
    input_gain = 0.01
    u_0 = beta_0 / (delta - beta_0)
    u_hc = beta_hc / (delta - beta_hc)
    u_sn = beta_sn / (delta - beta_sn)
    u = u_sn
    values: list[float] = []
    events: list[int] = []

    for _step in range(steps):
        events.append(int(u <= u_hc))
        if u <= u_0:
            u = decay_t0 * u + (1.0 - decay_t0) * f_0 + input_gain * current
        elif u <= u_hc:
            u_1 = (1.0 - alpha_t0) * u + alpha_t0 * f_0
            gap = beta_hc - delta * u_1 / (1.0 + u_1)
            inner = f_1
            if gap > 0.0:
                scale = math.exp(homoclinic_exponent * math.log(d * gap))
                inner = scale * (u_1 - f_1) + f_1
            u = inner + input_gain * current
        else:
            u = u_sn
        values.append(u)

    return {
        "spike_count": float(math.fsum(events)),
        "first_spike_step": float(
            next((index for index, event in enumerate(events, start=1) if event), -1)
        ),
        "final.u": values[-1],
        "min.u": min(values),
        "max.u": max(values),
        "mean.u": math.fsum(values) / len(values),
    }


def test_features_match_independent_section_4_iteration() -> None:
    """Committed features must match a fresh equation-level derivation."""
    spec = load_reference_trace_spec("medvedev_map_first_return_doi")
    expected = _independent_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "medvedev_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1016/j.physd.2005.01.021"
    assert expected["spike_count"] == 75.0
    assert expected["first_spike_step"] == 1.0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)


def test_committed_trace_validates_through_schema_runner() -> None:
    """The production schema runner must reproduce the feature contract."""
    spec = load_reference_trace_spec("medvedev_map_first_return_doi")
    report = validate_reference_trace_spec(spec)
    assert report.passed
    assert report.mismatches == ()
