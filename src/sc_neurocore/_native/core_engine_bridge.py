# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python ctypes bridge for core_engine Rust C-FFI

"""C-FFI bridge to Rust ``libcore_engine.so`` for SC arithmetic hot paths.

Provides:
- sc_multiply, sc_mux, sc_popcount, sc_popcount64
- sc_and_packed, sc_mux_packed, sc_popcount_packed
- sc_scc_packed, sc_cordiv_packed
- lfsr_create, lfsr_step, lfsr_encode, lfsr_destroy
"""

from __future__ import annotations

import ctypes as _ct
import pathlib as _pl
from typing import Any

_LIB_PATH = _pl.Path(__file__).parent / "libcore_engine.so"
_HAS_CORE_ENGINE = False
_lib: _ct.CDLL | None = None


def _get_lib() -> _ct.CDLL:
    """Return the loaded Rust core-engine library or raise if absent."""
    if _lib is None:
        raise RuntimeError(
            "libcore_engine.so is not loaded. Build the `core_engine` crate "
            "or ensure the .so is present next to this module."
        )
    return _lib


if _LIB_PATH.exists():
    try:
        _lib = _ct.CDLL(str(_LIB_PATH))

        # --- scalar ops ---
        _lib.sc_multiply.argtypes = [_ct.c_uint32, _ct.c_uint32]
        _lib.sc_multiply.restype = _ct.c_uint32

        _lib.sc_mux.argtypes = [_ct.c_uint32, _ct.c_uint32, _ct.c_uint32]
        _lib.sc_mux.restype = _ct.c_uint32

        _lib.sc_popcount.argtypes = [_ct.c_uint32]
        _lib.sc_popcount.restype = _ct.c_uint32

        _lib.sc_popcount64.argtypes = [_ct.c_uint64]
        _lib.sc_popcount64.restype = _ct.c_uint32

        _lib.sc_saturating_sub.argtypes = [_ct.c_uint32, _ct.c_uint32]
        _lib.sc_saturating_sub.restype = _ct.c_uint32

        # --- packed SIMD ops ---
        _lib.sc_and_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_and_packed.restype = None

        _lib.sc_mux_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_mux_packed.restype = None

        _lib.sc_popcount_packed.argtypes = [_ct.POINTER(_ct.c_uint64), _ct.c_size_t]
        _lib.sc_popcount_packed.restype = _ct.c_uint64

        _lib.sc_scc_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
            _ct.c_size_t,
        ]
        _lib.sc_scc_packed.restype = _ct.c_double

        _lib.sc_cordiv_packed.argtypes = [
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.POINTER(_ct.c_uint64),
            _ct.c_size_t,
        ]
        _lib.sc_cordiv_packed.restype = None

        # --- LFSR ---
        _lib.lfsr_create.argtypes = [_ct.c_uint16]
        _lib.lfsr_create.restype = _ct.c_void_p

        _lib.lfsr_step.argtypes = [_ct.c_void_p]
        _lib.lfsr_step.restype = _ct.c_uint16

        _lib.lfsr_encode.argtypes = [
            _ct.c_void_p,
            _ct.c_uint16,
            _ct.c_size_t,
            _ct.POINTER(_ct.POINTER(_ct.c_uint64)),
            _ct.POINTER(_ct.c_size_t),
        ]
        _lib.lfsr_encode.restype = None

        _lib.lfsr_destroy.argtypes = [_ct.c_void_p]
        _lib.lfsr_destroy.restype = None

        _lib.bitstream_free.argtypes = [_ct.POINTER(_ct.c_uint64), _ct.c_size_t]
        _lib.bitstream_free.restype = None

        _HAS_CORE_ENGINE = True
    except OSError:
        pass


def is_available() -> bool:
    """Return True if the Rust core engine is loaded."""
    return _HAS_CORE_ENGINE


def sc_multiply(a: int, b: int) -> int:
    """SC multiply: AND of two u32 bitstreams."""
    if not _HAS_CORE_ENGINE:
        return a & b
    return int(_get_lib().sc_multiply(_ct.c_uint32(a), _ct.c_uint32(b)))


def sc_mux(a: int, b: int, sel: int) -> int:
    """SC MUX: (sel & a) | (~sel & b)."""
    if not _HAS_CORE_ENGINE:
        return (sel & a) | (~sel & b) & 0xFFFFFFFF
    return int(_get_lib().sc_mux(_ct.c_uint32(a), _ct.c_uint32(b), _ct.c_uint32(sel)))


def sc_popcount(a: int) -> int:
    """Population count of a u32."""
    if not _HAS_CORE_ENGINE:
        return bin(a).count("1")
    return int(_get_lib().sc_popcount(_ct.c_uint32(a)))


def sc_popcount64(a: int) -> int:
    """Population count of a u64."""
    if not _HAS_CORE_ENGINE:
        return bin(a).count("1")
    return int(_get_lib().sc_popcount64(_ct.c_uint64(a)))


def sc_popcount_packed(data: list[int]) -> int:
    """Popcount of packed u64 array (from Python list)."""
    if not _HAS_CORE_ENGINE:
        return sum(bin(w).count("1") for w in data)
    n = len(data)
    arr = (_ct.c_uint64 * n)(*data)
    return int(_get_lib().sc_popcount_packed(arr, _ct.c_size_t(n)))


def sc_popcount_packed_np(data: Any) -> int:
    """Popcount of packed u64 numpy array (zero-copy)."""
    import numpy as np

    data = np.ascontiguousarray(data, dtype=np.uint64)
    ptr = data.ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return int(_get_lib().sc_popcount_packed(ptr, _ct.c_size_t(data.size)))


def sc_scc_packed(a: list[int], b: list[int], *, bit_length: int | None = None) -> float:
    """Stochastic cross-correlation of packed u64 arrays (from Python list)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    logical_bits = _logical_bit_length(n, bit_length)
    masked_a = _mask_trailing_words(a[:n], logical_bits)
    masked_b = _mask_trailing_words(b[:n], logical_bits)
    if not _HAS_CORE_ENGINE:
        return _python_scc_packed(masked_a, masked_b, bit_length=logical_bits)
    arr_a = (_ct.c_uint64 * n)(*masked_a)
    arr_b = (_ct.c_uint64 * n)(*masked_b)
    return float(
        _get_lib().sc_scc_packed(
            arr_a,
            arr_b,
            _ct.c_size_t(n),
            _ct.c_size_t(logical_bits),
        )
    )


def sc_scc_packed_np(a: Any, b: Any, *, bit_length: int | None = None) -> float:
    """Stochastic cross-correlation of packed u64 numpy arrays (zero-copy)."""
    import numpy as np

    a = np.ascontiguousarray(a, dtype=np.uint64)
    b = np.ascontiguousarray(b, dtype=np.uint64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    logical_bits = _logical_bit_length(n, bit_length)
    masked_a = np.ascontiguousarray(
        _mask_trailing_words(a[:n].tolist(), logical_bits), dtype=np.uint64
    )
    masked_b = np.ascontiguousarray(
        _mask_trailing_words(b[:n].tolist(), logical_bits), dtype=np.uint64
    )
    if not _HAS_CORE_ENGINE:
        return sc_scc_packed(masked_a.tolist(), masked_b.tolist(), bit_length=logical_bits)
    ptr_a = masked_a.ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    ptr_b = masked_b.ctypes.data_as(_ct.POINTER(_ct.c_uint64))
    return float(
        _get_lib().sc_scc_packed(
            ptr_a,
            ptr_b,
            _ct.c_size_t(n),
            _ct.c_size_t(logical_bits),
        )
    )


def lfsr_encode_packed(*, seed: int, threshold: int, bit_length: int) -> Any:
    """Encode an LFSR-16 stochastic bitstream as packed little-endian u64 words."""

    import numpy as np

    _validate_lfsr_contract(seed=seed, threshold=threshold, bit_length=bit_length)
    if not _HAS_CORE_ENGINE:
        return np.asarray(
            _python_lfsr_encode_packed(seed=seed, threshold=threshold, bit_length=bit_length),
            dtype=np.uint64,
        )

    lib = _get_lib()
    handle = lib.lfsr_create(_ct.c_uint16(seed & 0xFFFF))
    if not handle:
        raise RuntimeError("lfsr_create returned a null handle")
    out_ptr = _ct.POINTER(_ct.c_uint64)()
    out_words = _ct.c_size_t(0)
    try:
        lib.lfsr_encode(
            handle,
            _ct.c_uint16(threshold),
            _ct.c_size_t(bit_length),
            _ct.byref(out_ptr),
            _ct.byref(out_words),
        )
        word_count = int(out_words.value)
        if word_count != (bit_length + 63) // 64:
            raise RuntimeError("lfsr_encode returned an unexpected word count")
        if not out_ptr:
            raise RuntimeError("lfsr_encode returned a null bitstream pointer")
        data = np.ctypeslib.as_array(out_ptr, shape=(word_count,)).copy()
    finally:
        if out_ptr:
            lib.bitstream_free(out_ptr, _ct.c_size_t(int(out_words.value)))
        lib.lfsr_destroy(handle)
    return np.ascontiguousarray(_mask_trailing_words(data.tolist(), bit_length), dtype=np.uint64)


def lfsr_encode_bits(*, seed: int, threshold: int, bit_length: int) -> Any:
    """Encode an LFSR-16 stochastic bitstream as a uint8 0/1 numpy array."""

    import numpy as np

    packed = lfsr_encode_packed(seed=seed, threshold=threshold, bit_length=bit_length)
    bits = np.zeros(bit_length, dtype=np.uint8)
    for index in range(bit_length):
        bits[index] = (int(packed[index // 64]) >> (index % 64)) & 1
    return bits


def _logical_bit_length(word_count: int, bit_length: int | None) -> int:
    if bit_length is None:
        return word_count * 64
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    if bit_length > word_count * 64:
        raise ValueError("bit_length exceeds packed word capacity")
    return bit_length


def _mask_trailing_words(words: list[int], bit_length: int) -> list[int]:
    if not words:
        return []
    masked = [int(word) & 0xFFFF_FFFF_FFFF_FFFF for word in words]
    trailing = bit_length % 64
    if trailing:
        masked[-1] &= (1 << trailing) - 1
    return masked


def _validate_lfsr_contract(*, seed: int, threshold: int, bit_length: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= 0xFFFF:
        raise ValueError("seed must be a non-zero uint16")
    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or not 0 <= threshold <= 0xFFFF
    ):
        raise ValueError("threshold must fit uint16")
    if not isinstance(bit_length, int) or isinstance(bit_length, bool) or bit_length <= 0:
        raise ValueError("bit_length must be a positive integer")


def _python_lfsr_encode_packed(*, seed: int, threshold: int, bit_length: int) -> list[int]:
    words = [0 for _ in range((bit_length + 63) // 64)]
    state = seed & 0xFFFF
    for index in range(bit_length):
        if state < threshold:
            words[index // 64] |= 1 << (index % 64)
        feedback = ((state >> 15) ^ (state >> 13) ^ (state >> 12) ^ (state >> 10)) & 1
        state = ((state << 1) & 0xFFFF) | feedback
        if state == 0:
            state = 1
    return words


def _python_scc_packed(a: list[int], b: list[int], *, bit_length: int) -> float:
    n = float(bit_length)
    if n == 0.0:
        return 0.0

    a_count = sum(word.bit_count() for word in a)
    b_count = sum(word.bit_count() for word in b)
    and_count = sum((left & right).bit_count() for left, right in zip(a, b, strict=True))

    pa = a_count / n
    pb = b_count / n
    p_and = and_count / n
    numerator = p_and - (pa * pb)
    if abs(numerator) < 1e-12:
        return 0.0

    if numerator > 0.0:
        denominator = min(pa, pb) - (pa * pb)
    else:
        denominator = (pa * pb) - max(pa + pb - 1.0, 0.0)
    if abs(denominator) < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))
