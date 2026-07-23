# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAlignmentEnforcement from former test_array_guards.py

"""Focused suite: TestAlignmentEnforcement from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403

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
