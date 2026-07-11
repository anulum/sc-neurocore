# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map independent reference contract

"""Independent piecewise-map reference trace for the Rulkov map."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.reference_traces import load_reference_trace_spec
from tests.cosim_support import _rulkov_map_features


def test_rulkov_map_trace_features_match_independent_map_iteration() -> None:
    """Committed Rulkov features must match an independent piecewise-map iteration."""
    spec = load_reference_trace_spec("rulkov_map_driven_spiking_doi")

    expected = _rulkov_map_features(
        current=spec.protocol.inputs["I"],
        steps=spec.protocol.steps,
    )

    assert spec.schema_name == "rulkov_map"
    assert spec.provenance.kind == "map_iteration_reference"
    assert spec.provenance.citation == "doi:10.1103/PhysRevE.65.041922"
    assert spec.expected_features["spike_count"] > 0
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
