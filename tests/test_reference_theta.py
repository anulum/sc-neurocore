# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta reference-trace contracts

"""Theta reference-trace parity contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _theta_constant_current_features


def test_theta_trace_features_match_independent_phase_solution() -> None:
    """Committed theta features must match the tangent half-angle phase solution."""
    spec = load_reference_trace_spec("theta_constant_current_phase_analytic")

    expected = _theta_constant_current_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "theta"
    assert spec.provenance.kind == "analytic_closed_form"
    assert spec.provenance.citation == "doi:10.1137/0146017"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
