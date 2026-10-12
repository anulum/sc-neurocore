# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public packed-bitstream unpacking and refusal contracts

"""Exercise public unpacking shape, storage, output and nonconsuming refusal."""

from __future__ import annotations

import sys

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.vector_ops import unpack_bitstream as reference_unpack
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


@pytest.mark.parametrize("length", [0, 1, 7, 63, 64, 65, 128, 257])
def test_flat_unpack_matches_reference_and_preserves_readonly_storage(length: int) -> None:
    """Retain little-endian bit order, partial words and independent writable output."""
    words = np.array([0xACE1, 0xFEDCBA9876543210, 0, 2**64 - 1, 0x8001], dtype=np.uint64)
    words.setflags(write=False)
    before = words.tobytes()
    expected = reference_unpack(words, length)
    actual = engine.unpack_bitstream_numpy(words, length)
    generic = engine.unpack_bitstream(words.tolist(), length)
    assert generic == expected.tobytes()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (length,) and actual.dtype == np.uint8
    assert actual.flags.c_contiguous and actual.flags.writeable and actual.flags.owndata
    if length:
        actual[:] = 0
    assert words.tobytes() == before
    np.testing.assert_array_equal(engine.unpack_bitstream_numpy(words, length), expected)


@pytest.mark.parametrize("length", [0, 1, 64, 65])
@pytest.mark.parametrize("explicit_shape", [False, True])
def test_batched_unpack_retains_row_inference_and_shape_override(
    length: int, explicit_shape: bool
) -> None:
    """Compare every row with the public reference for empty, whole and partial words."""
    words = np.array([[0xACE1, 1], [0x8001, 0]], dtype=np.uint64)
    shape = (2, length) if explicit_shape else None
    total = 999 if explicit_shape else 2 * length
    expected = reference_unpack(words, total, original_shape=shape)
    actual = engine.unpack_bitstream(words.tolist(), total, original_shape=shape)
    assert actual == [row.tobytes() for row in expected]


def test_empty_storage_and_empty_batch_retain_their_declared_shapes() -> None:
    """Handle empty buffers and zero-row batches without inventing packed data."""
    empty = np.array([], dtype=np.uint64)
    assert engine.unpack_bitstream([], 0) == []
    assert engine.unpack_bitstream([0], 0) == b""
    assert engine.unpack_bitstream_numpy(empty, 0).shape == (0,)
    assert reference_unpack(empty, 0).shape == (0,)
    rows = np.empty((0, 0), dtype=np.uint64)
    assert engine.unpack_bitstream([], 0, original_shape=(0, 65)) == []
    assert reference_unpack(rows, 0, original_shape=(0, 65)).shape == (0, 65)
    assert reference_unpack(rows, 0).shape == (0, 0)


@pytest.mark.parametrize("length", [1, 65, 129])
def test_truncated_storage_is_refused_without_mutation_and_allows_retry(length: int) -> None:
    """Refuse a length extending beyond the available words in every unpack route."""
    words = np.zeros((length - 1) // 64, dtype=np.uint64)
    words.setflags(write=False)
    before = words.tobytes()
    with pytest.raises(ValueError, match="^packed words do not cover original_length$"):
        reference_unpack(words, length)
    with pytest.raises(ValueError, match="^packed words do not cover original_length$"):
        engine.unpack_bitstream_numpy(words, length)
    if len(words):
        with pytest.raises(ValueError, match="^packed words do not cover original_length$"):
            engine.unpack_bitstream(words.tolist(), length)
    else:
        with pytest.raises(ValueError, match="^packed words do not cover original_length$"):
            engine.unpack_bitstream([[]], length, original_shape=(1, length))
    assert words.tobytes() == before
    assert engine.unpack_bitstream([1], 1) == b"\x01"


def test_shape_batch_mismatch_and_ragged_short_row_do_not_publish_output() -> None:
    """Validate every requested row before returning a batch or modifying source words."""
    words = [[0, 0], [0]]
    with pytest.raises(ValueError, match="original_shape batch 3 does not match packed batch 2"):
        engine.unpack_bitstream(words, 130, original_shape=(3, 65))
    with pytest.raises(ValueError, match="^packed words do not cover original_length$"):
        engine.unpack_bitstream(words, 130, original_shape=(2, 65))
    assert words == [[0, 0], [0]]
    with pytest.raises(ValueError, match="original_shape batch 3 does not match packed batch 2"):
        reference_unpack(np.zeros((2, 2), dtype=np.uint64), 130, original_shape=(3, 65))


def test_unrepresentable_lengths_are_catchable_before_output_allocation() -> None:
    """Refuse unsigned lengths exceeding Python and NumPy dimension capacity."""
    words = np.array([0], dtype=np.uint64)
    length = sys.maxsize + 1
    for function in (reference_unpack, engine.unpack_bitstream_numpy):
        with pytest.raises(MemoryError, match="^unpacked bitstream cannot be allocated$"):
            function(words, length)
    with pytest.raises(MemoryError, match="^unpacked bitstream cannot be allocated$"):
        engine.unpack_bitstream([0], length)
    with pytest.raises(MemoryError, match="^unpacked bitstream cannot be allocated$"):
        engine.unpack_bitstream([], 0, original_shape=(0, length))
    assert engine.unpack_bitstream([1], 1) == b"\x01"


def test_negative_lengths_are_refused_before_any_output_exists() -> None:
    """Refuse a negative bit count in the reference and in the engine, each in its own way.

    The reference names the argument. The engine takes an unsigned count, so a
    negative one is refused at its boundary as an overflow before any work.
    """
    words = np.array([0], dtype=np.uint64)
    before = words.tobytes()
    with pytest.raises(ValueError, match="^original_length must be nonnegative$"):
        reference_unpack(words, -1)
    with pytest.raises(ValueError, match="^original_length must be nonnegative$"):
        reference_unpack(np.zeros((1, 1), dtype=np.uint64), 0, original_shape=(1, -1))
    with pytest.raises(ValueError, match="^original_length must be nonnegative$"):
        reference_unpack(np.zeros((2, 1), dtype=np.uint64), -2)
    with pytest.raises(OverflowError):
        engine.unpack_bitstream_numpy(words, -1)
    with pytest.raises(OverflowError):
        engine.unpack_bitstream([0], -1)
    with pytest.raises(OverflowError):
        engine.unpack_bitstream([[0]], 0, original_shape=(1, -1))
    assert words.tobytes() == before
    assert engine.unpack_bitstream([1], 1) == b"\x01"


def test_generic_strided_words_remain_supported_and_typed_layout_refuses() -> None:
    """Preserve generic sequence conversion and typed contiguous-input requirements."""
    words = np.array([1, 2, 3, 4], dtype=np.uint64)[::2]
    assert engine.unpack_bitstream(words, 65) == reference_unpack(words, 65).tobytes()
    with pytest.raises(ValueError, match="Cannot read numpy array"):
        engine.unpack_bitstream_numpy(words, 65)


@pytest.mark.parametrize("hint", [0, sys.maxsize])
def test_generic_unpack_uses_actual_words_without_reserving_the_length_hint(hint: int) -> None:
    """Unpack a valid short word sequence despite an inaccurate storage-length hint."""

    class Sequence:
        """Expose two real packed words independently of the declared length."""

        def __len__(self) -> int:
            return hint

        def __getitem__(self, index: int) -> int:
            return (1, 1)[index]

    source = Sequence()
    expected = b"\x01" + bytes(63) + b"\x01"
    assert engine.unpack_bitstream(source, 65) == expected
    assert engine.unpack_bitstream(source, 65) == expected


@pytest.mark.parametrize("dtype", [np.uint32, np.int64, np.float64])
def test_typed_unpack_retains_dtype_refusal_before_length_validation(
    dtype: type[np.uint32] | type[np.int64] | type[np.float64],
) -> None:
    """Keep the declared uint64 typed admission boundary even for invalid requested lengths."""
    words: npt.NDArray[np.uint32 | np.int64 | np.float64] = np.array([0], dtype=dtype)
    with pytest.raises(TypeError):
        engine.unpack_bitstream_numpy(words, sys.maxsize + 1)


def test_generic_word_conversion_preserves_its_public_value_error() -> None:
    """Refuse unsupported words without crashing or admitting partial output."""
    with pytest.raises(ValueError, match="Expected packed uint64 words as 1-D or 2-D sequence"):
        engine.unpack_bitstream(["bad"], 1)
    assert engine.unpack_bitstream([1], 1) == b"\x01"
