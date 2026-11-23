# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIntegrationBytes from former test_array_guards.py

"""Focused suite: TestIntegrationBytes from former test_array_guards.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore._native.array_guards import (
    require_c_contiguous,
)


class TestIntegrationBytes:
    """Byte access and buffer identity of guarded arrays."""

    def test_guarded_output_accepts_ctypes_style_access(self) -> None:
        """A guarded byte array exposes its bytes in order."""
        arr = np.arange(64, dtype=np.uint8)
        out = require_c_contiguous(arr, "stream", np.uint8)
        raw_bytes = out.tobytes()
        assert len(raw_bytes) == 64
        assert raw_bytes[0] == 0 and raw_bytes[63] == 63

    def test_guarded_output_pointer_stable_within_call(self) -> None:
        """A matching dtype keeps the caller's buffer.

        The guard must not re-alloc when dtype already matches — FFI
        callers assume the pointer handed in equals the pointer used.
        """
        arr = np.zeros(128, dtype=np.uint8)
        before = arr.__array_interface__["data"][0]
        out = require_c_contiguous(arr, "stream", np.uint8)
        after = out.__array_interface__["data"][0]
        assert before == after, "no-copy promise violated for matching dtype"

    def test_dtype_coerce_returns_fresh_buffer_by_pointer(self) -> None:
        """A dtype cast hands back a new buffer."""
        arr = np.zeros(16, dtype=np.int64)
        out = require_c_contiguous(arr, "stream", np.uint8)
        assert arr.__array_interface__["data"][0] != out.__array_interface__["data"][0], (
            "dtype cast must produce a new buffer"
        )
