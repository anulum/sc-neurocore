# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNdarrayHappyPath from former test_array_guards.py

"""Focused suite: TestNdarrayHappyPath from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403


class TestNdarrayHappyPath:
    def test_contiguous_matching_dtype_returns_same_object(self):
        arr = np.zeros(16, dtype=np.uint8)
        out = require_c_contiguous(arr, "x", np.uint8)
        assert out is arr, "matching dtype should be identity return (no copy)"

    def test_contiguous_no_dtype_check_returns_same_object(self):
        arr = np.arange(32, dtype=np.float32)
        out = require_c_contiguous(arr, "x")
        assert out is arr
        assert out.dtype == np.float32

    def test_2d_contiguous_matching_dtype(self):
        arr = np.zeros((4, 8), dtype=np.int32)
        out = require_c_contiguous(arr, "matrix", np.int32)
        assert out is arr
        assert out.shape == (4, 8)

    def test_empty_array_is_contiguous(self):
        arr = np.empty(0, dtype=np.uint8)
        out = require_c_contiguous(arr, "empty", np.uint8)
        assert out is arr
        assert out.size == 0

    def test_scalar_shape_zero_d(self):
        arr = np.asarray(42, dtype=np.int64)
        out = require_c_contiguous(arr, "scalar", np.int64)
        assert out is arr
        assert out.ndim == 0
