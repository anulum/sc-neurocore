# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored Studio graph-envelope validation contracts

"""Exercise malformed envelopes through the public graph import and export APIs."""

from __future__ import annotations

import pytest

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.network_graph import (
    GraphEnvelopeRefusal,
    envelope_to_graph,
    graph_to_envelope,
)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ([], "Network graph payload must be an object"),
        ({"nodes": []}, "Graph envelope nodes must be an object"),
        ({"edges": {}}, "Graph envelope edges must be a list"),
        ({"nodes": {"": {}}}, "Graph envelope node ids must be non-empty strings"),
        ({"nodes": {"a": 1}}, "Graph envelope node 'a' must be an object"),
        ({"edges": [1]}, "Graph envelope edge 0 must be an object"),
        (
            {"edges": [{"source": "a", "target": None}]},
            "Graph envelope edge 0 target must be a non-empty string",
        ),
    ],
)
def test_envelope_import_uses_explicit_authored_refusals(payload: object, reason: str) -> None:
    """Each structural rule retains its sentence and ValueError compatibility."""
    with pytest.raises(GraphEnvelopeRefusal) as refusal:
        envelope_to_graph(payload)
    assert str(refusal.value) == reason
    assert isinstance(refusal.value, AuthoredRefusal)
    assert isinstance(refusal.value, ValueError)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ([], "Network graph must be an object"),
        ({"populations": []}, "Invalid network graph: Network has no populations"),
    ],
)
def test_envelope_export_uses_explicit_authored_refusals(payload: object, reason: str) -> None:
    """The public exporter preserves schema refusals without trusting plain ValueError."""
    with pytest.raises(GraphEnvelopeRefusal) as refusal:
        graph_to_envelope(payload)
    assert str(refusal.value) == reason
    assert isinstance(refusal.value, AuthoredRefusal)
