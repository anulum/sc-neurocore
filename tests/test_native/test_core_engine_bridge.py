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
        bit_length: ct.c_size_t,
    ) -> float:
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
        return ct.c_void_p(int(seed.value) or 1)

    def lfsr_encode(
        self,
        _ptr: ct.c_void_p,
        threshold: ct.c_uint16,
        length: ct.c_uint32,
        out_ptr: ct.POINTER(ct.POINTER(ct.c_uint64)),
        out_words: ct.POINTER(ct.c_size_t),
    ) -> None:
        bits = int(length.value)
        words = (bits + 63) // 64
        array_type = ct.c_uint64 * words
        arr = array_type()
        for index in range(bits):
            if index < int(threshold.value):
                arr[index // 64] |= 1 << (index % 64)
        self._last_array = arr
        out_ptr_cast = ct.cast(out_ptr, ct.POINTER(ct.POINTER(ct.c_uint64)))
        out_words_cast = ct.cast(out_words, ct.POINTER(ct.c_size_t))
        out_ptr_cast[0] = ct.cast(arr, ct.POINTER(ct.c_uint64))
        out_words_cast[0] = words

    def bitstream_free(self, _ptr: ct.POINTER(ct.c_uint64), _size: ct.c_size_t) -> None:
        return None

    def lfsr_destroy(self, _ptr: ct.c_void_p) -> None:
        return None


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
    assert ceb.sc_scc_packed([0xAAAAAAAAAAAAAAAA], [0xAAAAAAAAAAAAAAAA]) == pytest.approx(1.0)
    assert ceb.sc_scc_packed([0xAAAAAAAAAAAAAAAA], [0x5555555555555555]) == pytest.approx(-1.0)
    assert ceb.sc_scc_packed([0xFFFFFFFFFFFFFFFF], [0xFFFFFFFFFFFFFFFF]) == pytest.approx(0.0)


def test_python_scc_fallback_honors_logical_bit_length() -> None:
    ceb._HAS_CORE_ENGINE = False

    assert ceb.sc_scc_packed([0b1010_1010], [0b1010_1010], bit_length=8) == pytest.approx(1.0)
    assert ceb.sc_scc_packed([0b1010_1010], [0b0101_0101], bit_length=8) == pytest.approx(-1.0)

    masked = ceb.sc_scc_packed([0xFFFF_FFFF_FFFF_FF0F], [0xFFFF_FFFF_FFFF_FF0F], bit_length=8)
    assert masked == pytest.approx(1.0)


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

    masked_scc = ceb.sc_scc_packed(
        [0xFFFFFFFFFFFFFFFF],
        [0xFFFFFFFFFFFFFFFF],
        bit_length=17,
    )
    assert masked_scc == pytest.approx(0.25)

    scc_np = ceb.sc_scc_packed_np(
        np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64),
        np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64),
    )
    assert scc_np == pytest.approx(1.0)


def test_native_lfsr_encode_packed_delegates_to_loaded_library() -> None:
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = _FakeCoreLib()

    words = ceb.lfsr_encode_packed(seed=0xACE1, threshold=4, bit_length=8)

    assert words.dtype == np.uint64
    assert words.tolist() == [0b1111]


def test_lfsr_encode_bits_rejects_invalid_contract_values() -> None:
    with pytest.raises(ValueError, match="seed"):
        ceb.lfsr_encode_bits(seed=0, threshold=1, bit_length=8)

    with pytest.raises(ValueError, match="threshold"):
        ceb.lfsr_encode_bits(seed=1, threshold=70000, bit_length=8)

    with pytest.raises(ValueError, match="bit_length"):
        ceb.lfsr_encode_bits(seed=1, threshold=1, bit_length=0)


def test_sc_scc_packed_empty_returns_zero() -> None:
    assert ceb.sc_scc_packed([], []) == 0.0


def test_logical_bit_length_validation_guards() -> None:
    with pytest.raises(ValueError, match="bit_length must be positive"):
        ceb._logical_bit_length(2, 0)
    with pytest.raises(ValueError, match="exceeds packed word capacity"):
        ceb._logical_bit_length(1, 1000)


def test_python_scc_packed_degenerate_branches() -> None:
    # A zero bit_length short-circuits to a zero correlation.
    assert ceb._python_scc_packed([], [], bit_length=0) == 0.0
    # An under-counted bit_length pushes pa>1 so the denominator collapses to 0.
    assert ceb._python_scc_packed([0b11], [0b01], bit_length=1) == 0.0


def test_lfsr_encode_bits_unpacks_packed_words() -> None:
    bits = ceb.lfsr_encode_bits(seed=1, threshold=0x8000, bit_length=80)
    assert bits.shape == (80,)
    assert {int(value) for value in np.unique(bits)}.issubset({0, 1})


def test_lfsr_and_scc_python_fallbacks_when_engine_absent() -> None:
    # The autouse fixture restores _HAS_CORE_ENGINE afterwards.
    ceb._HAS_CORE_ENGINE = False
    packed = ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=128)
    assert packed.dtype == np.uint64
    correlation = ceb.sc_scc_packed_np(np.asarray(packed), np.asarray(packed), bit_length=128)
    assert -1.0 <= correlation <= 1.0


def test_lfsr_encode_packed_rejects_null_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NullHandleLib:
        @staticmethod
        def lfsr_create(seed: object) -> int:
            return 0

    ceb._HAS_CORE_ENGINE = True
    monkeypatch.setattr(ceb, "_get_lib", lambda: _NullHandleLib())
    with pytest.raises(RuntimeError, match="lfsr_create returned a null handle"):
        ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=64)
