# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGPUVecAnd from former test_gpu_backend.py

"""Focused suite: TestGPUVecAnd from former test_gpu_backend.py."""

from __future__ import annotations

from tests.gpu_backend_support import *  # noqa: F403


class TestGPUVecAnd:
    def test_identity(self):
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0xAAAAAAAAAAAAAAAA], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0xAAAAAAAAAAAAAAAA

    def test_zero(self):
        a = xp.array([0xFFFFFFFFFFFFFFFF], dtype=xp.uint64)
        b = xp.array([0], dtype=xp.uint64)
        result = gpu_vec_and(a, b)
        assert int(to_host(result)[0]) == 0
