# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC packed input and refusal contracts

"""Exercise packed storage and error atomicity through the installed engine."""

from __future__ import annotations

import subprocess
import sys
from array import array
from typing import Generic, TypeVar

import pytest
from sc_neurocore_engine import BitStreamTensor, HDCVector


HDCItem = TypeVar("HDCItem")


class HDCInputSequence(Generic[HDCItem]):
    """Provide actual Python sequence items with a separately declared length."""

    def __init__(self, items: tuple[HDCItem, ...], declared_length: int | None) -> None:
        """Retain the items and the length returned by the Python protocol."""
        self.items = items
        self.declared_length = declared_length

    def __len__(self) -> int:
        """Return the declared length or report unavailable length information."""
        if self.declared_length is None:
            raise ValueError("sequence length unavailable")
        return self.declared_length

    def __getitem__(self, index: int) -> HDCItem:
        """Return a stored item or raise the sequence's terminal IndexError."""
        return self.items[index]


def test_constructor_refuses_zero_dimension() -> None:
    """Zero-dimensional random construction raises an ordinary Python error."""
    with pytest.raises(ValueError, match="^bitstream length must be > 0$"):
        BitStreamTensor(0)
    with pytest.raises(ValueError, match="^bitstream length must be > 0$"):
        HDCVector(0)


def test_packed_constructor_refuses_zero_length() -> None:
    """Packed construction refuses zero length before examining its words."""
    with pytest.raises(ValueError, match="^bitstream length must be > 0$"):
        BitStreamTensor.from_packed([], 0)


@pytest.mark.parametrize(
    ("data", "length"), [([], 1), ([], 65), ([0], 65), ([0, 0], 64), ([0, 0, 0], 65)]
)
def test_packed_word_count_matches_logical_length(data: list[int], length: int) -> None:
    """Short and excessive storage fail before a tensor is admitted."""
    with pytest.raises(ValueError, match=r"^packed word count must equal ceil\(length / 64\)$"):
        BitStreamTensor.from_packed(data, length)
    with pytest.raises(ValueError, match=r"^packed word count must equal ceil\(length / 64\)$"):
        HDCVector.from_packed(data, length)


@pytest.mark.parametrize(
    ("data", "length"), [([2], 1), ([1 << 63], 63), ([0, 2], 65), ([0, 1 << 63], 127)]
)
def test_packed_padding_must_be_zero(data: list[int], length: int) -> None:
    """Unused high bits cannot inflate logical popcount or normalized distance."""
    with pytest.raises(ValueError, match="^unused packed bits must be zero$"):
        BitStreamTensor.from_packed(data, length)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128])
def test_canonical_words_preserve_distance_and_copy_ownership(length: int) -> None:
    """Each logical bit counts once across full and partial final words."""
    full, remainder = divmod(length, 64)
    words = [(1 << 64) - 1] * full + ([(1 << remainder) - 1] if remainder else [])
    original = words.copy()
    tensor = BitStreamTensor.from_packed(words, length)
    zero = BitStreamTensor.from_packed([0] * len(words), length)
    words[0] = 0
    detached = tensor.data
    detached[0] = 0
    assert tensor.data == original
    assert tensor.popcount() == length
    assert tensor.hamming_distance(zero) == 1.0
    assert tensor.hamming_distance(tensor) == 0.0
    assert len(tensor) == length


@pytest.mark.parametrize("operation", ["xor", "xor_inplace", "hamming_distance"])
@pytest.mark.parametrize(("first_length", "second_length"), [(63, 64), (64, 65), (65, 129)])
def test_binary_length_refusal_preserves_both_tensors(
    operation: str, first_length: int, second_length: int
) -> None:
    """Unequal lengths refuse before either tensor's packed state changes."""
    first = BitStreamTensor(first_length, seed=1)
    second = BitStreamTensor(second_length, seed=2)
    before = (first.data, second.data, first.length, second.length)
    message = "Hamming distance" if operation == "hamming_distance" else "XOR"
    with pytest.raises(ValueError, match=f"^bitstream lengths must match for {message}$"):
        getattr(first, operation)(second)
    assert (first.data, second.data, first.length, second.length) == before


def test_bundle_refuses_empty_input() -> None:
    """The native bundle API refuses an empty vector sequence without panicking."""
    with pytest.raises(ValueError, match=r"^Cannot bundle zero vectors\.$"):
        BitStreamTensor.bundle([])
    with pytest.raises(ValueError, match=r"^Cannot bundle zero vectors\.$"):
        HDCVector.bundle([])


@pytest.mark.parametrize("lengths", [(63, 64), (64, 65), (65, 64), (65, 64, 65)])
def test_bundle_refuses_mismatched_lengths_without_changing_inputs(
    lengths: tuple[int, ...],
) -> None:
    """Bundle checks every logical length before reading any packed word."""
    vectors = [BitStreamTensor(length, seed=index + 1) for index, length in enumerate(lengths)]
    before = [vector.data for vector in vectors]
    with pytest.raises(ValueError, match="^bitstream lengths must match for bundle$"):
        BitStreamTensor.bundle(vectors)
    assert [vector.data for vector in vectors] == before


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_partial_word_bundle_matches_strict_majority(count: int) -> None:
    """Valid bundles retain strict-majority algebra and zero padding."""
    words = [(0b1101, 1), (0b1011, 0), (0b1001, 1), (0b0110, 0), (0b0011, 1)][:count]
    tensors = [BitStreamTensor.from_packed(list(data), 65) for data in words]
    expected = [
        sum(
            (sum((data[index] >> bit) & 1 for data in words) > count // 2) << bit
            for bit in range(64)
        )
        for index in range(2)
    ]
    result = BitStreamTensor.bundle(tensors)
    assert result.data == expected
    assert result.length == 65


def test_hdc_vector_operators_propagate_native_length_refusals() -> None:
    """The vector facade carries the same refusals through its operators."""
    first = HDCVector(64, seed=1)
    second = HDCVector(65, seed=2)
    with pytest.raises(ValueError, match="^bitstream lengths must match for XOR$"):
        _ = first * second
    with pytest.raises(ValueError, match="^bitstream lengths must match for bundle$"):
        _ = first + second
    with pytest.raises(ValueError, match="^bitstream lengths must match for Hamming distance$"):
        first.similarity(second)


def test_packed_rotation_and_xor_preserve_partial_word_algebra() -> None:
    """Rotation crosses the word boundary while XOR preserves zero padding."""
    first = BitStreamTensor.from_packed([1, 1], 65)
    second = BitStreamTensor.from_packed([2, 0], 65)
    first.rotate_right(1)
    assert first.data == [3, 0]
    expected = first.xor(second)
    assert expected.data == [1, 0]
    assert first.xor_inplace(second) is None
    assert first.data == expected.data


def test_random_constructor_reports_real_reservation_failure() -> None:
    """An impossible native reservation returns MemoryError in a live consumer."""
    program = """
import sys
from sc_neurocore_engine import BitStreamTensor
try:
    BitStreamTensor(sys.maxsize * 2 + 1, seed=1)
except MemoryError as error:
    assert str(error) == 'cannot allocate packed bitstream'
else:
    raise AssertionError('impossible packed allocation was accepted')
assert BitStreamTensor.from_packed([1], 1).popcount() == 1
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("operation", ["packed", "bundle"])
def test_sequence_extraction_reports_real_capacity_failure(operation: str) -> None:
    """An impossible Python sequence hint cannot abort the native consumer."""
    program = """
import sys
from sc_neurocore_engine import BitStreamTensor
class EmptySequence:
    def __len__(self): return sys.maxsize
    def __getitem__(self, index): raise IndexError(index)
try:
    if sys.argv[1] == 'packed':
        BitStreamTensor.from_packed(EmptySequence(), 1)
    else:
        BitStreamTensor.bundle(EmptySequence())
except MemoryError as error:
    assert str(error) == 'cannot allocate HDC input sequence'
else:
    raise AssertionError('impossible sequence allocation was accepted')
assert BitStreamTensor.from_packed([1], 1).popcount() == 1
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, operation],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("declared_length", [0, 1, 2, None])
def test_sequence_length_hint_does_not_replace_actual_words(declared_length: int | None) -> None:
    """Short, excess and unavailable hints retain actual sequence iteration."""
    sequence = HDCInputSequence((1, 1), declared_length)
    value = BitStreamTensor.from_packed(sequence, 65)
    assert value.data == [1, 1] and value.popcount() == 2


@pytest.mark.parametrize("container", [tuple, array, memoryview])
def test_standard_word_sequences_keep_their_packed_bits(container: object) -> None:
    """Tuple, array and buffer-view inputs retain the original u64 sequence contract."""
    words = (1, 1)
    if container is tuple:
        source: object = words
    elif container is array:
        source = array("Q", words)
    else:
        source = memoryview(array("Q", words))
    tensor = BitStreamTensor.from_packed(source, 65)
    assert tensor.data == [1, 1] and tensor.length == 65


@pytest.mark.parametrize("declared_length", [0, 1, 2, None])
def test_bundle_length_hint_does_not_replace_actual_tensors(declared_length: int | None) -> None:
    """Tensor sequence iteration preserves majority algebra independently of its hint."""
    first = BitStreamTensor.from_packed([1, 1], 65)
    second = BitStreamTensor.from_packed([3, 0], 65)
    sequence = HDCInputSequence((first, second), declared_length)
    result = BitStreamTensor.bundle(sequence)
    assert result.data == [1, 0] and result.length == 65
    assert first.data == [1, 1] and second.data == [3, 0]


@pytest.mark.parametrize("source", [None, {}, "words", 1])
def test_extraction_keeps_non_sequence_and_string_type_errors(source: object) -> None:
    """Both HDC entry points refuse the original non-sequence and string inputs."""
    with pytest.raises(TypeError):
        BitStreamTensor.from_packed(source, 65)
    with pytest.raises(TypeError):
        BitStreamTensor.bundle(source)


@pytest.mark.parametrize("word", [-1, 1 << 64])
def test_packed_word_extraction_keeps_unsigned_overflow_errors(word: int) -> None:
    """Out-of-range unsigned words retain the native integer conversion refusal."""
    with pytest.raises(OverflowError):
        BitStreamTensor.from_packed([word], 1)
