# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_core_engine_bridge.py

from __future__ import annotations

"""Support extracted from test_core_engine_bridge.py."""

from collections.abc import Generator


import ctypes as ct


from pathlib import Path


import runpy


from typing import Any, cast


import numpy as np


import pytest


from sc_neurocore._native import core_engine_bridge as ceb


class _FakeCoreLib:
    """ctypes-compatible stand-in for the loaded Rust core engine."""

    def __init__(self) -> None:
        self._last_array: Any = None

    def sc_multiply(self, a: ct.c_uint32, b: ct.c_uint32) -> int:
        """Return the bitwise SC multiplication result."""
        return int(a.value & b.value)

    def sc_mux(self, a: ct.c_uint32, b: ct.c_uint32, sel: ct.c_uint32) -> int:
        """Return the bitwise SC mux result."""
        return int(((sel.value & a.value) | ((~sel.value) & b.value)) & 0xFFFFFFFF)

    def sc_popcount(self, a: ct.c_uint32) -> int:
        """Return the popcount of a u32 wrapper."""
        return int(a.value).bit_count()

    def sc_popcount64(self, a: ct.c_uint64) -> int:
        """Return the popcount of a u64 wrapper."""
        return int(a.value).bit_count()

    def sc_popcount_packed(self, ptr: Any, size: ct.c_size_t) -> int:
        """Return the popcount across a packed pointer slice."""
        return sum(int(ptr[i]).bit_count() for i in range(int(size.value)))

    def sc_scc_packed(
        self,
        a_ptr: Any,
        b_ptr: Any,
        size: ct.c_size_t,
        bit_length: ct.c_size_t,
    ) -> float:
        """Return a deterministic packed-SCC result for native delegation tests."""
        n = int(size.value)
        bits = int(bit_length.value)
        if n == 0:
            return 0.0
        if bits != n * 64:
            return 0.25
        same = 0
        total = 0
        for i in range(n):
            a = int(a_ptr[i])
            b = int(b_ptr[i])
            same += (~(a ^ b) & ((1 << 64) - 1)).bit_count()
            total += 64
        return same / total

    def lfsr_create(self, seed: ct.c_uint16) -> ct.c_void_p:
        """Return a non-null opaque LFSR handle."""
        return ct.c_void_p(int(seed.value) or 1)

    def lfsr_encode(
        self,
        _ptr: ct.c_void_p,
        threshold: ct.c_uint16,
        length: ct.c_size_t,
        out_ptr: Any,
        out_words: Any,
    ) -> None:
        """Write a deterministic packed bitstream into the output pointers."""
        bits = int(length.value)
        words = (bits + 63) // 64
        array_type = ct.c_uint64 * words
        arr = array_type()
        for index in range(bits):
            if index < int(threshold.value):
                arr[index // 64] |= 1 << (index % 64)
        self._last_array = arr
        out_ptr_cast = cast(Any, ct.cast(out_ptr, ct.POINTER(ct.POINTER(ct.c_uint64))))
        out_words_cast = cast(Any, ct.cast(out_words, ct.POINTER(ct.c_size_t)))
        out_ptr_cast[0] = ct.cast(arr, ct.POINTER(ct.c_uint64))
        out_words_cast[0] = words

    def bitstream_free(self, _ptr: Any, _size: ct.c_size_t) -> None:
        """Release hook for parity with the C-FFI surface."""
        return None

    def lfsr_destroy(self, _ptr: ct.c_void_p) -> None:
        """Destroy hook for parity with the C-FFI surface."""
        return None


@pytest.fixture(autouse=True)
def _restore_core_bridge_state() -> Generator[None, None, None]:
    """Restore the module-level native bridge state after each test."""
    old_has = ceb._HAS_CORE_ENGINE
    old_lib = ceb._lib
    try:
        yield
    finally:
        ceb._HAS_CORE_ENGINE = old_has
        ceb._lib = old_lib


__all__ = [
    "Generator",
    "ct",
    "Path",
    "runpy",
    "Any",
    "cast",
    "np",
    "pytest",
    "ceb",
    "_FakeCoreLib",
    "_restore_core_bridge_state",
]
