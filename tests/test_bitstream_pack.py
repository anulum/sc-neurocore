# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public generic and typed bitstream packing contracts

"""Exercise packing bit order, Python sequence protocols and typed admission."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel.vector_ops import pack_bitstream as reference_pack
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


@pytest.mark.parametrize(
    "length", [0, 1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024, 1031]
)
@pytest.mark.parametrize("readonly", [False, True])
def test_typed_pack_retains_nonzero_byte_semantics_and_owns_output(
    length: int, readonly: bool
) -> None:
    """Retain exact SIMD boundaries and tails with independent, writable output."""
    storage = np.resize(np.array([0, 1, 2, 127, 128, 255], dtype=np.uint8), length + 1)
    bits = storage[1:]
    bits.setflags(write=not readonly)
    before = storage.tobytes()
    normalized = np.asarray(bits != 0, dtype=np.uint8)
    expected = reference_pack(normalized)
    packed_bytes = np.packbits(normalized, bitorder="little").tobytes()
    padded_bytes = packed_bytes + bytes((-len(packed_bytes)) % 8)
    independent = np.frombuffer(padded_bytes, dtype="<u8")
    actual = engine.pack_bitstream_numpy(bits)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, independent)
    np.testing.assert_array_equal(actual, engine.pack_bitstream(bits.tolist()))
    assert actual.shape == ((length + 63) // 64,) and actual.dtype == np.uint64
    assert actual.flags.owndata and actual.flags.c_contiguous and actual.flags.writeable
    assert not np.shares_memory(actual, bits)
    np.testing.assert_array_equal(engine.unpack_bitstream_numpy(actual, length), normalized)
    actual[:] = 0
    assert storage.tobytes() == before
    np.testing.assert_array_equal(engine.pack_bitstream_numpy(bits), expected)


@pytest.mark.parametrize("layout", ["strided", "reversed"])
def test_typed_pack_refuses_noncontiguous_input_without_consuming_it(layout: str) -> None:
    """Keep layout errors catchable and permit a subsequent valid public call."""
    storage = np.array([0, 1, 2, 0, 128, 0, 255, 0], dtype=np.uint8)
    bits = storage[::2] if layout == "strided" else storage[::-1]
    bits.setflags(write=False)
    before = storage.tobytes()
    with pytest.raises(ValueError, match="Cannot read numpy array"):
        engine.pack_bitstream_numpy(bits)
    assert storage.tobytes() == before
    np.testing.assert_array_equal(
        engine.pack_bitstream_numpy(storage), np.array([86], dtype=np.uint64)
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.int8, np.float64])
def test_typed_pack_refuses_other_dtypes_without_coercion(
    dtype: type[np.uint16] | type[np.int8] | type[np.float64],
) -> None:
    """Preserve the uint8 admission boundary and caller values after refusal."""
    bits: npt.NDArray[np.uint16 | np.int8 | np.float64] = np.array([0, 1], dtype=dtype)
    before = bits.tobytes()
    with pytest.raises(TypeError):
        engine.pack_bitstream_numpy(bits)
    assert bits.tobytes() == before


@pytest.mark.parametrize("shape", [(), (1, 2), (1, 1, 2)])
def test_typed_pack_refuses_other_ranks_without_mutating_input(shape: tuple[int, ...]) -> None:
    """Retain one-dimensional typed admission for scalars and batched arrays."""
    bits = np.ones(shape, dtype=np.uint8)
    before = bits.tobytes()
    with pytest.raises(TypeError):
        engine.pack_bitstream_numpy(bits)
    assert bits.tobytes() == before


@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 256, 257])
@pytest.mark.parametrize(
    "kind", ["list", "tuple", "bytes", "bytearray", "memoryview", "array", "strided", "reversed"]
)
def test_generic_pack_preserves_sequence_formats_and_independent_lists(
    length: int, kind: str
) -> None:
    """Retain generic sequence admission, nonzero-byte semantics and empty outputs."""
    values = np.resize(np.array([0, 1, 2, 128, 255], dtype=np.uint8), length)
    source: object
    if kind == "list":
        source = values.tolist()
    elif kind == "tuple":
        source = tuple(values.tolist())
    elif kind == "bytes":
        source = values.tobytes()
    elif kind == "bytearray":
        source = bytearray(values.tobytes())
    elif kind == "memoryview":
        source = memoryview(values.tobytes())
    elif kind == "strided":
        storage = np.repeat(values, 2)
        source = storage[::2]
    elif kind == "reversed":
        source = values[::-1]
        values = values[::-1]
    else:
        source = values
    if isinstance(source, np.ndarray):
        source.setflags(write=False)
    before = (
        bytes(source)
        if isinstance(source, (bytes, bytearray, memoryview))
        else np.asarray(source, dtype=np.uint8).tobytes()
    )
    normalized = np.asarray(values != 0, dtype=np.uint8)
    expected = reference_pack(normalized).tolist()
    actual = engine.pack_bitstream(source)
    assert isinstance(actual, list) and actual == expected
    if actual:
        actual[0] = 0
    after = (
        bytes(source)
        if isinstance(source, (bytes, bytearray, memoryview))
        else np.asarray(source, dtype=np.uint8).tobytes()
    )
    assert after == before
    if isinstance(source, np.ndarray):
        assert not source.flags.writeable
    assert engine.pack_bitstream(source) == expected


@pytest.mark.parametrize("kind", ["list", "tuple", "buffers", "array"])
def test_generic_pack_retains_batched_and_ragged_row_shapes(kind: str) -> None:
    """Return a separate Python word list for each row, including empty rows."""
    rows = [[], [0, 1, 255], [255] * 65]
    source: object
    if kind == "tuple":
        source = tuple(tuple(row) for row in rows)
    elif kind == "buffers":
        source = [b"", bytearray(rows[1]), memoryview(bytes(rows[2]))]
    elif kind == "array":
        rows = [[0, 1, 255], [255, 0, 128]]
        source = np.array(rows, dtype=np.uint8)
    else:
        source = rows
    before = repr(source)
    expected = [reference_pack(np.asarray(row, dtype=np.uint8) != 0).tolist() for row in rows]
    actual = engine.pack_bitstream(source)
    assert actual == expected and all(isinstance(row, list) for row in actual)
    actual[-1][0] = 0
    assert repr(source) == before
    assert engine.pack_bitstream(source) == expected


@pytest.mark.parametrize("hint", [0, 1, 1024, sys.maxsize])
def test_generic_pack_uses_actual_sequence_items_with_inexact_lengths(hint: int) -> None:
    """Retain the index-based sequence protocol without substituting its length hint."""

    class Sequence:
        """An indexable input whose declared length differs from its actual contents."""

        def __len__(self) -> int:
            return hint

        def __getitem__(self, index: int) -> int:
            return (1, 0, 255)[index]

    source = Sequence()
    assert engine.pack_bitstream(source) == [5]
    assert engine.pack_bitstream(source) == [5]


def test_generic_byte_buffer_subclasses_retain_buffer_extraction() -> None:
    """Preserve PyO3's raw byte-buffer fallback when subclass iteration refuses."""

    class Bytes(bytes):
        """Immutable byte storage with a refusing custom iterator."""

        def __iter__(self) -> Iterator[int]:
            raise ValueError("custom iteration unavailable")

    class ByteArray(bytearray):
        """Mutable byte storage with a refusing custom iterator."""

        def __iter__(self) -> Iterator[int]:
            raise ValueError("custom iteration unavailable")

    assert engine.pack_bitstream(Bytes(b"\x00\x01\xff")) == [6]
    assert engine.pack_bitstream(ByteArray(b"\x00\x01\xff")) == [6]


@pytest.mark.parametrize("source", ["bad", {0: 1}, {0, 1}, [-1], [256], [0.5], [[[1]]]])
def test_generic_pack_preserves_invalid_input_value_error(source: object) -> None:
    """Retain conversion refusal and permit subsequent valid packing."""
    before = repr(source)
    with pytest.raises(ValueError, match="^Expected a 1-D or 2-D array of uint8 bits\\.$"):
        engine.pack_bitstream(source)
    assert repr(source) == before
    assert engine.pack_bitstream([0, 1, 255]) == [6]


def test_generic_pack_refuses_an_iterator_without_consuming_it() -> None:
    """Retain sequence-only admission rather than silently exhausting generators."""
    source = iter([0, 1, 255])
    with pytest.raises(ValueError, match="Expected a 1-D or 2-D array of uint8 bits"):
        engine.pack_bitstream(source)
    assert list(source) == [0, 1, 255]
