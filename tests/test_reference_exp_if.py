# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF independent reference-trace contract

"""Independent source-equation reference parity for ExpIF."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sc_neurocore.neurons.reference_traces import ReferenceTraceSpec, load_reference_trace_spec
from tests.cosim_support import _exp_if_rk4_features

_PARITY_CASES: list[tuple[str, str, str, str, Callable[[ReferenceTraceSpec], dict[str, float]]]] = [
    (
        "exp_if_driven_rk4_doi",
        "exp_if",
        "independent_rk4_reference",
        "doi:10.1523/JNEUROSCI.23-37-11628.2003",
        lambda spec: _exp_if_rk4_features(
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
    """Reproduce every committed feature independently to ``1e-12``."""
    spec = load_reference_trace_spec(trace_name)
    expected = reference(spec)

    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
