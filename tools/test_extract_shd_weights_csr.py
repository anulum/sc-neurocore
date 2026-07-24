# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCsr from former test_extract_shd_weights.py

"""Focused suite: TestCsr from former test_extract_shd_weights.py."""

from __future__ import annotations

from extract_shd_weights_support import *  # noqa: F403


class TestCsr:
    def test_dense_matrix(self) -> None:
        # 90% sparse: 1 nz out of 10
        w = torch.zeros(2, 5, dtype=torch.int8)
        w[0, 2] = 5
        w[1, 4] = -3
        csr = to_csr(w)
        assert csr["shape"] == [2, 5]
        assert csr["nnz"] == 2
        assert csr["row_ptr"] == [0, 1, 2]
        assert csr["col_idx"] == [2, 4]
        assert csr["values"] == [5, -3]
        assert csr["sparsity_pct"] == pytest.approx(80.0)

    def test_all_zero(self) -> None:
        w = torch.zeros(3, 4, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 0
        assert csr["row_ptr"] == [0, 0, 0, 0]
        assert csr["col_idx"] == []
        assert csr["values"] == []

    def test_dense_no_zeros(self) -> None:
        w = torch.full((2, 3), 5, dtype=torch.int8)
        csr = to_csr(w)
        assert csr["nnz"] == 6
        assert csr["sparsity_pct"] == 0.0
        assert csr["row_ptr"] == [0, 3, 6]
