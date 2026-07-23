# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSparsifyTopK from former test_federated_sc.py

"""Focused suite: TestSparsifyTopK from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestSparsifyTopK:
    def test_top_k_selects_largest(self):
        g = np.array([0.1, -0.5, 0.3, -0.8, 0.2])
        sparse, mask = sparsify_topk(g, k=2)
        assert mask[1] == 1  # |-0.5| is 2nd largest
        assert mask[3] == 1  # |-0.8| is largest
        assert np.count_nonzero(mask) == 2

    def test_sparse_preserves_values(self):
        g = np.array([0.1, 0.5, 0.3])
        sparse, mask = sparsify_topk(g, k=1)
        idx = np.argmax(mask)
        assert sparse[idx] == g[idx]

    def test_zero_entries(self):
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sparse, mask = sparsify_topk(g, k=2)
        assert np.count_nonzero(sparse) == 2

    def test_k_exceeds_length(self):
        g = np.array([1.0, 2.0])
        sparse, mask = sparsify_topk(g, k=10)
        np.testing.assert_array_almost_equal(sparse, g)
