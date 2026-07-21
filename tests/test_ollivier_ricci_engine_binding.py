# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ollivier-Ricci engine-binding contracts

"""Installed-extension contracts for the Ollivier-Ricci binding."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.math import topology

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _complete(node_count: int) -> NDArray[np.float64]:
    graph = np.ones((node_count, node_count), dtype=np.float64)
    np.fill_diagonal(graph, 0.0)
    return graph


def _direct(graph: NDArray[np.float64], i: int, j: int) -> float:
    return float(
        extension.py_ollivier_ricci_curvature(graph.ravel().tolist(), graph.shape[0], i, j)
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_ollivier_ricci_curvature

    assert function.__name__ == "py_ollivier_ricci_curvature"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == "(knm_flat, n, i, j)"
    assert engine.py_ollivier_ricci_curvature is function
    assert "py_ollivier_ricci_curvature" in engine.__all__


def test_complete_disconnected_and_self_edge_contracts() -> None:
    assert _direct(_complete(4), 0, 1) == pytest.approx(2.0 / 3.0)

    disconnected = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert _direct(disconnected, 0, 2) == 0.0
    assert _direct(_complete(4), 2, 2) == 0.0


@pytest.mark.parametrize(
    ("knm_flat", "n", "i", "j", "message"),
    (
        ([], 0, 0, 0, "knm must be a square coupling matrix with at least one node"),
        ([0.0, 1.0], 2, 0, 1, "knm must be a square coupling matrix with at least one node"),
        ([0.0, -1.0, 1.0, 0.0], 2, 0, 1, "knm must contain only finite, non-negative values"),
        (
            [0.0, float("nan"), 1.0, 0.0],
            2,
            0,
            1,
            "knm must contain only finite, non-negative values",
        ),
        ([0.0, 1.0, 1.0, 0.0], 2, 0, 2, "node index out of range for coupling graph"),
    ),
)
def test_validation_errors_preserve_exact_value_error_contract(
    knm_flat: list[float], n: int, i: int, j: int, message: str
) -> None:
    with pytest.raises(ValueError, match=rf"^{message}$"):
        extension.py_ollivier_ricci_curvature(knm_flat, n, i, j)


def test_production_rust_backend_uses_the_installed_extension() -> None:
    graph = _complete(4)
    direct = _direct(graph, 0, 1)

    assert topology._HAS_RUST_TOPOLOGY is True
    assert topology._rust_ollivier is engine.py_ollivier_ricci_curvature
    assert topology.ollivier_ricci_curvature(graph, 0, 1, backend="rust") == direct
