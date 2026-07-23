# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonNdarrayNonContiguousRejected from former test_array_guards.py

"""Focused suite: TestNonNdarrayNonContiguousRejected from former test_array_guards.py."""

from __future__ import annotations

from array_guards_support import *  # noqa: F403

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
