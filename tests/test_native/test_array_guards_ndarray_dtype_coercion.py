# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNdarrayDtypeCoercion from former test_array_guards.py

"""Focused suite: TestNdarrayDtypeCoercion from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403

class TestNdarrayDtypeCoercion:
    def test_wrong_dtype_triggers_astype(self):
        arr = np.arange(8, dtype=np.int64)
        out = require_c_contiguous(arr, "x", np.uint8)
        assert out is not arr, "dtype mismatch must cast"
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, np.arange(8, dtype=np.uint8))

    def test_wrong_dtype_preserves_values(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = require_c_contiguous(arr, "x", np.float32)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_dtype_alias_np_dtype_object(self):
        """``dtype`` parameter accepts both np.generic subclasses and np.dtype instances."""
        arr = np.zeros(4, dtype=np.uint8)
        out_type = require_c_contiguous(arr, "x", np.uint8)
        out_dtype = require_c_contiguous(arr, "x", np.dtype("uint8"))
        assert out_type is arr
        assert out_dtype is arr
