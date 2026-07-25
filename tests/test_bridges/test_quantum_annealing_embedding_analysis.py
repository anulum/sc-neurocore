# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing embedding-analysis contracts

from __future__ import annotations


import pytest

from sc_neurocore.bridges.quantum_annealing import (
    EmbeddingAnalyzer,
    IsingModel,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising


def test_embedding_analyzer_sparse_and_dense_models() -> None:
    """Embedding estimates report graph density and chain capacity."""
    sparse = EmbeddingAnalyzer().analyze(simple_ising())
    assert sparse["n_logical_qubits"] == 3
    assert sparse["n_couplers"] == 2
    assert sparse["min_chain_estimate"] == 1
    dense = IsingModel(
        J={(first, second): -1.0 for first in range(5) for second in range(first + 1, 5)},
        n_qubits=5,
    )
    assert EmbeddingAnalyzer().analyze(dense)["density"] == 1.0
    with pytest.raises(ValueError, match="non-empty"):
        EmbeddingAnalyzer().analyze(IsingModel())
