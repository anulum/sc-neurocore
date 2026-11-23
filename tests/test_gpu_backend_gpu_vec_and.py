# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPUVecAnd from former test_gpu_backend.py

"""Focused suite: TestGPUVecAnd from former test_gpu_backend.py."""

from __future__ import annotations

from sc_neurocore.accel.gpu_backend import (
    gpu_vec_and,
    to_host,
    xp,
)


class TestGPUVecAnd:
    """Bitwise AND of word vectors."""

    def test_identity(self) -> None:
        """AND with all ones keeps the word."""
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0xAAAAAAAAAAAAAAAA], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0xAAAAAAAAAAAAAAAA

    def test_zero(self) -> None:
        """AND with zero gives zero."""
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0
