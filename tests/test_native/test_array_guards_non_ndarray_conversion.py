# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonNdarrayConversion from former test_array_guards.py

"""Focused suite: TestNonNdarrayConversion from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403


class TestNonNdarrayConversion:
    def test_python_list_converted_and_contiguous(self):
        out = require_c_contiguous([1, 2, 3, 4], "x", np.uint8)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, [1, 2, 3, 4])

    def test_python_tuple_converted(self):
        out = require_c_contiguous((10, 20, 30), "x", np.int32)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, [10, 20, 30])

    def test_nested_list_converted_2d(self):
        out = require_c_contiguous([[1, 2], [3, 4]], "mat", np.float32)
        assert out.shape == (2, 2)
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_list_no_dtype_keeps_numpy_default(self):
        out = require_c_contiguous([1, 2, 3], "x")
        assert isinstance(out, np.ndarray)
        assert out.flags["C_CONTIGUOUS"]

    def test_empty_list_converts_to_empty_array(self):
        out = require_c_contiguous([], "empty", np.uint8)
        assert isinstance(out, np.ndarray)
        assert out.size == 0
        assert out.dtype == np.uint8
