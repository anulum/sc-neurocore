# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNdarrayNonContiguousRejected from former test_array_guards.py

"""Focused suite: TestNdarrayNonContiguousRejected from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403


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
