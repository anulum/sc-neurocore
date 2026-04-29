# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for Studio graph JSON inputs

"""Property-based fuzz tests for Studio graph JSON and NIR input boundaries."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.studio.network_graph import graph_to_nir, nir_to_graph, validate_graph

_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=12), children, max_size=4)
    ),
    max_leaves=24,
)


@given(payload=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_validate_graph_never_crashes_on_json_payloads(payload: object) -> None:
    errors = validate_graph(payload)
    assert isinstance(errors, list)
    assert all(isinstance(error, str) and error for error in errors)


@given(payload=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_graph_to_nir_is_structured_or_rejected(payload: object) -> None:
    try:
        nir = graph_to_nir(payload)
    except ValueError as exc:
        assert str(exc)
        return

    assert nir["format"] == "nir"
    assert isinstance(nir["nodes"], dict)
    assert isinstance(nir["edges"], list)


@given(payload=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_nir_to_graph_is_structured_or_rejected(payload: object) -> None:
    try:
        graph = nir_to_graph(payload)
    except ValueError as exc:
        assert str(exc)
        return

    assert isinstance(graph["populations"], list)
    assert isinstance(graph["projections"], list)
    assert validate_graph(graph) == [] or graph["populations"] == []


def test_validate_graph_reports_malformed_population_and_projection() -> None:
    errors = validate_graph(
        {
            "populations": [{"count": 4}, "bad-population"],
            "projections": [{"id": "p0", "source": "missing"}, "bad-projection"],
        }
    )
    assert any("id must be a non-empty string" in error for error in errors)
    assert any("must be an object" in error for error in errors)
    assert any("target must be a non-empty string" in error for error in errors)


@pytest.mark.parametrize(
    "payload",
    [
        {"nodes": [], "edges": []},
        {"nodes": {"": {}}, "edges": []},
        {"nodes": {"a": []}, "edges": []},
        {"nodes": {"a": {}}, "edges": [{}]},
        {"nodes": {"a": {}}, "edges": ["bad-edge"]},
    ],
)
def test_nir_to_graph_rejects_malformed_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        nir_to_graph(payload)
