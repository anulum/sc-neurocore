# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quadratic IF reference-trace contracts

"""Quadratic IF reference-trace parity contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _quadratic_if_zero_current_features


def test_quadratic_if_trace_features_match_independent_analytic_solution() -> None:
    """Committed QIF features must match the analytic zero-current Riccati flow."""
    spec = load_reference_trace_spec("quadratic_if_zero_current_analytic")

    expected = _quadratic_if_zero_current_features(
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "quadratic_if"
    assert spec.provenance.citation == "doi:10.1152/jn.2000.83.2.808"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
