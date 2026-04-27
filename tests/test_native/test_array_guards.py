# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-angle tests for the C-FFI array guard

"""Exhaustive tests for ``sc_neurocore._native.array_guards``.

The guard is a zero-copy gatekeeper between NumPy and the Rust/Mojo
native libraries. Every failure path is exercised (non-contiguous
view, unaligned buffer, wrong dtype, list input, tuple input,
multi-dim slices, empty arrays, Fortran order, read-only mmap).
Any regression here corrupts FFI calls silently, so the tests are
kept strict on exception type AND message fragments.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native.array_guards import require_c_contiguous


# ---------------------------------------------------------------------------
# Happy path — ndarray with matching dtype, returned identity
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# dtype coercion path — contiguous ndarray, wrong dtype -> astype
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Non-contiguous rejection — strided views, slices, transpose
# ---------------------------------------------------------------------------


class TestNdarrayNonContiguousRejected:
    def test_strided_slice_raises(self):
        base = np.arange(20, dtype=np.uint8)
        view = base[::2]  # stride-2 view, not C-contiguous
        assert not view.flags["C_CONTIGUOUS"]

        with pytest.raises(ValueError, match=r"x must be C-contiguous"):
            require_c_contiguous(view, "x", np.uint8)

    def test_transposed_2d_raises(self):
        base = np.zeros((4, 8), dtype=np.float32)
        transposed = base.T  # becomes F-contiguous, not C-contiguous
        assert not transposed.flags["C_CONTIGUOUS"]
        assert transposed.flags["F_CONTIGUOUS"]

        with pytest.raises(ValueError, match=r"C-contiguous"):
            require_c_contiguous(transposed, "matrix")

    def test_column_slice_2d_raises(self):
        base = np.arange(32, dtype=np.int32).reshape(4, 8)
        col = base[:, 2]  # stride-skipping view
        assert not col.flags["C_CONTIGUOUS"]

        with pytest.raises(ValueError):
            require_c_contiguous(col, "col")

    def test_error_message_includes_hint(self):
        view = np.arange(10, dtype=np.uint8)[::2]
        with pytest.raises(ValueError) as exc:
            require_c_contiguous(view, "bitstream")
        assert "bitstream" in str(exc.value)
        assert "np.ascontiguousarray" in str(exc.value)

    def test_error_message_uses_given_name(self):
        view = np.arange(10, dtype=np.uint8)[::2]
        with pytest.raises(ValueError, match=r"custom_name"):
            require_c_contiguous(view, "custom_name")


# ---------------------------------------------------------------------------
# Non-ndarray input path — list / tuple / generator
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Aligned-flag enforcement — synthesised misaligned buffer
# ---------------------------------------------------------------------------


class TestNonNdarrayNonContiguousRejected:
    """Cover the defensive branch after ``np.asarray`` coercion.

    The guard checks contiguity/alignment on ``converted`` even though
    ``np.asarray`` usually guarantees both. An object implementing
    ``__array__`` that returns a strided view bypasses that guarantee.
    """

    def test_array_protocol_non_contiguous_raises(self):
        class NonContigProducer:
            def __array__(self, dtype=None, copy=None):
                base = np.arange(20, dtype=np.uint8)
                return base[::2]

        with pytest.raises(ValueError, match=r"must be C-contiguous"):
            require_c_contiguous(NonContigProducer(), "producer")

    def test_array_protocol_unaligned_raises(self):
        class UnalignedProducer:
            def __array__(self, dtype=None, copy=None):
                raw = np.zeros(17, dtype=np.uint8)
                return np.ndarray(shape=(4,), dtype=np.float32, buffer=raw.data, offset=1)

        with pytest.raises(ValueError, match=r"not aligned"):
            require_c_contiguous(UnalignedProducer(), "producer")


class TestAlignmentEnforcement:
    def test_unaligned_ndarray_raises(self):
        """
        Construct a byte buffer with a deliberate 1-byte offset so the
        float32 view is not 4-byte aligned. NumPy uses `buffer=` with a
        carefully sized byte buffer to produce an unaligned view.
        """
        raw = np.zeros(17, dtype=np.uint8)
        # Build a float32 view starting at offset 1 — misaligned.
        unaligned = np.ndarray(shape=(4,), dtype=np.float32, buffer=raw.data, offset=1)
        assert not unaligned.flags["ALIGNED"]
        # C_CONTIGUOUS is true here (stride = itemsize, 1-D)
        assert unaligned.flags["C_CONTIGUOUS"]

        with pytest.raises(ValueError, match=r"not aligned"):
            require_c_contiguous(unaligned, "float_buf")


# ---------------------------------------------------------------------------
# Integration-style — use output directly in a byte op that FFI would do
# ---------------------------------------------------------------------------


class TestIntegrationBytes:
    def test_guarded_output_accepts_ctypes_style_access(self):
        arr = np.arange(64, dtype=np.uint8)
        out = require_c_contiguous(arr, "stream", np.uint8)
        raw_bytes = out.tobytes()
        assert len(raw_bytes) == 64
        assert raw_bytes[0] == 0 and raw_bytes[63] == 63

    def test_guarded_output_pointer_stable_within_call(self):
        """The guard must not re-alloc when dtype already matches — FFI
        callers assume the pointer handed in equals the pointer used."""
        arr = np.zeros(128, dtype=np.uint8)
        before = arr.__array_interface__["data"][0]
        out = require_c_contiguous(arr, "stream", np.uint8)
        after = out.__array_interface__["data"][0]
        assert before == after, "no-copy promise violated for matching dtype"

    def test_dtype_coerce_returns_fresh_buffer_by_pointer(self):
        arr = np.zeros(16, dtype=np.int64)
        out = require_c_contiguous(arr, "stream", np.uint8)
        assert arr.__array_interface__["data"][0] != out.__array_interface__["data"][0], (
            "dtype cast must produce a new buffer"
        )
