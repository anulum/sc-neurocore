# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for NIR import inputs

"""Property-based fuzz tests for package-level NIR import boundaries."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir


@given(payload=st.binary(max_size=512))
@settings(max_examples=80, deadline=None)
def test_fuzz_from_nir_rejects_malformed_files(payload: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.nir"
        path.write_bytes(payload)

        try:
            network = from_nir(path)
        except ValueError as exc:
            assert str(exc)
            return

        assert hasattr(network, "nodes")


@given(
    src=st.text(max_size=8),
    dst=st.text(max_size=8),
    dt=st.one_of(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0),
        st.integers(min_value=-10, max_value=10),
    ),
)
@settings(max_examples=100, deadline=None)
def test_fuzz_from_nir_rejects_malformed_edges_and_dt(src: str, dst: str, dt: float) -> None:
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([1])}),
            "output": nir.Output(output_type={"output": np.array([1])}),
        },
        edges=[(src, dst)],
        type_check=False,
    )

    try:
        network = from_nir(graph, dt=dt)
    except ValueError as exc:
        assert str(exc)
        return

    assert src == "input"
    assert dst == "output"
    assert dt > 0
    assert list(network.nodes) == ["input", "output"]


def test_from_nir_reports_missing_edge_endpoint() -> None:
    graph = nir.NIRGraph(
        nodes={"input": nir.Input(input_type={"input": np.array([1])})},
        edges=[("input", "missing")],
        type_check=False,
    )

    with pytest.raises(ValueError, match="destination 'missing' not found"):
        from_nir(graph)
