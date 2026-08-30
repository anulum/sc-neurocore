# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator independent reference contract

"""Independent analytic reference trace for the Perfect Integrator."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _perfect_integrator_sawtooth_features


def test_perfect_integrator_trace_features_match_independent_sawtooth_solution() -> None:
    """Committed perfect-integrator features must match the exact reset sawtooth."""
    spec = load_reference_trace_spec("perfect_integrator_constant_current_sawtooth")

    expected = _perfect_integrator_sawtooth_features(
        current=spec.protocol.inputs["I"],
        dt=spec.protocol.dt,
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "perfect_integrator"
    assert spec.provenance.kind == "analytic_exact_integral"
    assert spec.provenance.citation == "doi:10.1007/978-94-007-3858-4_6, section 1.1"
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
