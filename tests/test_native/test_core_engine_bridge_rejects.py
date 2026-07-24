# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects) from former test_core_engine_bridge.py

from __future__ import annotations

from core_engine_bridge_support import *  # noqa: F403


def test_lfsr_encode_bits_rejects_invalid_contract_values() -> None:
    """Reject invalid public LFSR bitstream contract values."""
    with pytest.raises(ValueError, match="seed"):
        ceb.lfsr_encode_bits(seed=0, threshold=1, bit_length=8)

    with pytest.raises(ValueError, match="threshold"):
        ceb.lfsr_encode_bits(seed=1, threshold=70000, bit_length=8)

    with pytest.raises(ValueError, match="bit_length"):
        ceb.lfsr_encode_bits(seed=1, threshold=1, bit_length=0)


def test_logical_bit_length_validation_guards() -> None:
    """Reject impossible logical bit-length requests."""
    with pytest.raises(ValueError, match="bit_length must be positive"):
        ceb._logical_bit_length(2, 0)
    with pytest.raises(ValueError, match="exceeds packed word capacity"):
        ceb._logical_bit_length(1, 1000)


def test_lfsr_encode_packed_rejects_null_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject native LFSR creation failures before reading output pointers."""

    class _NullHandleLib:
        """Library stub returning a null LFSR handle."""

        @staticmethod
        def lfsr_create(seed: object) -> int:
            """Return a null handle regardless of seed."""
            return 0

    ceb._HAS_CORE_ENGINE = True
    monkeypatch.setattr(ceb, "_get_lib", lambda: _NullHandleLib())
    with pytest.raises(RuntimeError, match="lfsr_create returned a null handle"):
        ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=64)


def test_lfsr_encode_packed_rejects_unexpected_word_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject native LFSR outputs whose word count cannot represent the request."""

    class _WrongWordCountLib(_FakeCoreLib):
        """Library stub reporting the wrong number of output words."""

        def lfsr_encode(
            self,
            _ptr: ct.c_void_p,
            _threshold: ct.c_uint16,
            _length: ct.c_size_t,
            out_ptr: Any,
            out_words: Any,
        ) -> None:
            """Write one word while reporting two words."""
            array_type = ct.c_uint64 * 1
            arr = array_type(0xFFFF)
            self._last_array = arr
            out_ptr_cast = cast(Any, ct.cast(out_ptr, ct.POINTER(ct.POINTER(ct.c_uint64))))
            out_words_cast = cast(Any, ct.cast(out_words, ct.POINTER(ct.c_size_t)))
            out_ptr_cast[0] = ct.cast(arr, ct.POINTER(ct.c_uint64))
            out_words_cast[0] = 2

    ceb._HAS_CORE_ENGINE = True
    monkeypatch.setattr(ceb, "_get_lib", lambda: _WrongWordCountLib())

    with pytest.raises(RuntimeError, match="unexpected word count"):
        ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=64)


def test_lfsr_encode_packed_rejects_null_bitstream_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject native LFSR calls that report words without a data pointer."""

    class _NullBitstreamLib(_FakeCoreLib):
        """Library stub reporting a word count but no output pointer."""

        def lfsr_encode(
            self,
            _ptr: ct.c_void_p,
            _threshold: ct.c_uint16,
            _length: ct.c_size_t,
            _out_ptr: Any,
            out_words: Any,
        ) -> None:
            """Leave the output pointer null while reporting one output word."""
            out_words_cast = cast(Any, ct.cast(out_words, ct.POINTER(ct.c_size_t)))
            out_words_cast[0] = 1

    ceb._HAS_CORE_ENGINE = True
    monkeypatch.setattr(ceb, "_get_lib", lambda: _NullBitstreamLib())

    with pytest.raises(RuntimeError, match="null bitstream pointer"):
        ceb.lfsr_encode_packed(seed=1, threshold=0x8000, bit_length=64)
