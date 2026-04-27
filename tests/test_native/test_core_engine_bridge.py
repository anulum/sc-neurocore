# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage for core_engine ctypes bridge branches

from __future__ import annotations

import ctypes as ct

import numpy as np
import pytest

from sc_neurocore._native import core_engine_bridge as ceb


class _FakeCoreLib:
    def sc_multiply(self, a: ct.c_uint32, b: ct.c_uint32) -> int:
        return int(a.value & b.value)

    def sc_mux(self, a: ct.c_uint32, b: ct.c_uint32, sel: ct.c_uint32) -> int:
        return int(((sel.value & a.value) | ((~sel.value) & b.value)) & 0xFFFFFFFF)

    def sc_popcount(self, a: ct.c_uint32) -> int:
        return int(a.value).bit_count()

    def sc_popcount64(self, a: ct.c_uint64) -> int:
        return int(a.value).bit_count()

    def sc_popcount_packed(self, ptr: ct.POINTER(ct.c_uint64), size: ct.c_size_t) -> int:
        return sum(int(ptr[i]).bit_count() for i in range(int(size.value)))

    def sc_scc_packed(
        self,
        a_ptr: ct.POINTER(ct.c_uint64),
        b_ptr: ct.POINTER(ct.c_uint64),
        size: ct.c_size_t,
    ) -> float:
        n = int(size.value)
        if n == 0:
            return 0.0
        same = 0
        total = 0
        for i in range(n):
            a = int(a_ptr[i])
            b = int(b_ptr[i])
            same += (~(a ^ b) & ((1 << 64) - 1)).bit_count()
            total += 64
        return same / total


@pytest.fixture(autouse=True)
def _restore_core_bridge_state() -> None:
    old_has = ceb._HAS_CORE_ENGINE
    old_lib = ceb._lib
    try:
        yield
    finally:
        ceb._HAS_CORE_ENGINE = old_has
        ceb._lib = old_lib


def test_get_lib_raises_when_unloaded() -> None:
    ceb._lib = None
    with pytest.raises(RuntimeError, match="not loaded"):
        ceb._get_lib()


def test_scalar_fallbacks_when_native_absent() -> None:
    ceb._HAS_CORE_ENGINE = False
    assert ceb.is_available() is False
    assert ceb.sc_multiply(0b1010, 0b1100) == 0b1000
    assert ceb.sc_mux(0xAAAAAAAA, 0x55555555, 0xFFFFFFFF) == 0xAAAAAAAA
    assert ceb.sc_popcount(0b101101) == 4
    assert ceb.sc_popcount64((1 << 63) | 0b101) == 3
    assert ceb.sc_popcount_packed([0b1010, 0b1111]) == 6
    assert ceb.sc_scc_packed([0xF0F0], [0xF0F0]) == 0.0


def test_numpy_scc_empty_input_returns_zero() -> None:
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = _FakeCoreLib()
    assert ceb.sc_scc_packed_np(np.array([], dtype=np.uint64), np.array([], dtype=np.uint64)) == 0.0


def test_native_paths_delegate_to_loaded_library() -> None:
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = _FakeCoreLib()

    assert ceb.is_available() is True
    assert ceb.sc_multiply(0b1010, 0b1100) == 0b1000
    assert ceb.sc_mux(0xAAAAAAAA, 0x55555555, 0xFFFFFFFF) == 0xAAAAAAAA
    assert ceb.sc_popcount(0b101101) == 4
    assert ceb.sc_popcount64((1 << 63) | 0b101) == 3
    assert ceb.sc_popcount_packed([0b1010, 0b1111]) == 6

    packed = np.array([0xF0F0F0F0F0F0F0F0, 0xAAAAAAAAAAAAAAAA], dtype=np.uint64)
    assert ceb.sc_popcount_packed_np(packed) == 64

    scc = ceb.sc_scc_packed([0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF])
    assert scc == pytest.approx(1.0)

    scc_np = ceb.sc_scc_packed_np(
        np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64),
        np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64),
    )
    assert scc_np == pytest.approx(1.0)
