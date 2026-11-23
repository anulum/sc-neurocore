# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAllToAll from former test_topology_generators.py

"""Focused suite: TestAllToAll from former test_topology_generators.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.network.topology import (
    all_to_all,
)
from tests.topology_generators_support import (
    _validate_csr,
)


class TestAllToAll:
    """The all-to-all connectivity generator."""

    def test_csr_valid(self) -> None:
        """The generated graph is a valid CSR matrix."""
        indptr, indices, data = all_to_all(10, 10, weight=0.5)
        _validate_csr(indptr, indices, data, 10, 10)

    def test_full_matrix(self) -> None:
        """Every pair is connected."""
        n = 8
        indptr, indices, data = all_to_all(n, n, weight=1.0)
        assert len(indices) == n * n

    def test_rectangular(self) -> None:
        """A rectangular graph connects every pair."""
        indptr, indices, data = all_to_all(5, 10, weight=0.3)
        _validate_csr(indptr, indices, data, 5, 10)
        assert len(indices) == 50

    def test_weight_value(self) -> None:
        """Every edge has the given weight."""
        _, _, data = all_to_all(5, 5, weight=0.42)
        np.testing.assert_allclose(data, 0.42)
