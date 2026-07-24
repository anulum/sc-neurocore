# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (native_delegation) from former test_core_engine_bridge.py

from __future__ import annotations

from core_engine_bridge_support import *  # noqa: F403

def test_native_paths_delegate_to_loaded_library() -> None:
    """Delegate scalar, packed, and NumPy operations to a loaded core engine."""
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = cast(ct.CDLL, _FakeCoreLib())

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
    """Delegate packed LFSR generation to the loaded native library."""
    ceb._HAS_CORE_ENGINE = True
    ceb._lib = cast(ct.CDLL, _FakeCoreLib())

    words = ceb.lfsr_encode_packed(seed=0xACE1, threshold=4, bit_length=8)

    assert words.dtype == np.uint64
    assert words.tolist() == [0b1111]


def test_lfsr_encode_bits_unpacks_packed_words() -> None:
    """Unpack generated packed words into a uint8 bit vector."""
    bits = ceb.lfsr_encode_bits(seed=1, threshold=0x8000, bit_length=80)
    assert bits.shape == (80,)
    assert {int(value) for value in np.unique(bits)}.issubset({0, 1})


def test_sc_scc_packed_empty_returns_zero() -> None:
    """Return zero for empty list-backed packed SCC inputs."""
    assert ceb.sc_scc_packed([], []) == 0.0


def test_mask_trailing_words_empty_input_is_stable() -> None:
    """Return an empty mask result for empty packed-word inputs."""
    assert ceb._mask_trailing_words([], 64) == []


def test_python_scc_packed_degenerate_branches() -> None:
    """Cover degenerate pure-Python SCC denominator branches."""
    # A zero bit_length short-circuits to a zero correlation.
    assert ceb._python_scc_packed([], [], bit_length=0) == 0.0
    # An under-counted bit_length pushes pa>1 so the denominator collapses to 0.
    assert ceb._python_scc_packed([0b11], [0b01], bit_length=1) == 0.0
