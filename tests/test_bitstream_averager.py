# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public sliding-window bitstream estimator contracts

"""Exercise estimator traces and nonconsuming refusals through public classes."""

from __future__ import annotations

import copy
import pickle
import sys

import numpy as np
import pytest

from sc_neurocore.utils.bitstreams import BitstreamAverager
from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine


@pytest.mark.parametrize("window", [256, 1024])
@pytest.mark.parametrize("numpy_samples", [False, True])
def test_large_window_keeps_an_integer_sum_across_full_wraps(
    window: int, numpy_samples: bool
) -> None:
    """Retain the binary mean above uint8 capacity for Python and NumPy input samples."""
    reference = BitstreamAverager(window=window)
    native = engine.BitstreamAverager(window=window)
    for _ in range(2 * window):
        bit = np.uint8(1) if numpy_samples else 1
        reference.push(bit)
        native.push(bit)
        assert reference.estimate() == native.estimate() == 1.0
    for count in range(1, window + 1):
        bit = np.uint8(0) if numpy_samples else 0
        reference.push(bit)
        native.push(bit)
        assert reference.estimate() == native.estimate() == (window - count) / window
    reference.reset()
    native.reset()
    assert reference.estimate() == native.estimate() == 0.0


@pytest.mark.parametrize("window", [1, 2, 4, 65])
@pytest.mark.parametrize("bits", [(), (1,), (0, 1, 1, 0, 1, 0, 0, 1) * 20])
def test_python_and_native_follow_the_observed_window(window: int, bits: tuple[int, ...]) -> None:
    """Compare every partial, full and wrapped estimate with the binary sample mean."""
    reference = BitstreamAverager(window=window)
    native = engine.BitstreamAverager(window=window)
    assert native.window == reference.window == window
    assert native.estimate() == reference.estimate() == 0.0
    observed: list[int] = []
    for bit in bits:
        observed.append(bit)
        reference.push(bit)
        assert native.push(bit) is None
        retained = observed[-window:]
        expected = sum(retained) / len(retained)
        assert native.estimate() == reference.estimate() == expected
    for estimator in (reference, native):
        assert estimator.reset() is None
        assert estimator.estimate() == 0.0
        assert estimator.window == window
        estimator.push(1)
        assert estimator.estimate() == 1.0


def test_zero_window_is_rejected_before_an_estimator_is_admitted() -> None:
    """Refuse the empty window at both public constructors without aborting."""
    with pytest.raises(ValueError, match="^window must be > 0$"):
        BitstreamAverager(window=0)
    with pytest.raises(ValueError, match="^window must be > 0$"):
        engine.BitstreamAverager(window=0)
    assert engine.BitstreamAverager(window=1).estimate() == 0.0


def test_impossible_buffer_allocation_is_catchable_and_allows_construction() -> None:
    """Request real unallocatable storage, then create a small usable estimator."""
    for factory in (BitstreamAverager, engine.BitstreamAverager):
        with pytest.raises(MemoryError, match="^Bitstream averager buffer cannot be allocated$"):
            factory(window=sys.maxsize)
        estimator = factory(window=4)
        estimator.push(1)
        assert estimator.estimate() == 1.0
        estimator.reset()
        assert estimator.estimate() == 0.0


@pytest.mark.parametrize("prefix", [(1,), (1, 0, 1, 1, 0)])
@pytest.mark.parametrize("invalid_bit", [2, 127, 255])
def test_invalid_binary_sample_does_not_consume_a_window_position(
    prefix: tuple[int, ...], invalid_bit: int
) -> None:
    """Preserve partial and wrapped state through refusal and subsequent valid samples."""
    reference = BitstreamAverager(window=4)
    native = engine.BitstreamAverager(window=4)
    observed = list(prefix)
    for bit in prefix:
        reference.push(bit)
        native.push(bit)
    before = sum(observed[-4:]) / min(len(observed), 4)
    for estimator in (reference, native):
        with pytest.raises(ValueError, match="^Bit must be 0 or 1\\.$"):
            estimator.push(invalid_bit)
        assert estimator.estimate() == before
    for bit in (0, 1, 0, 0, 1):
        observed.append(bit)
        reference.push(bit)
        native.push(bit)
        retained = observed[-4:]
        assert reference.estimate() == native.estimate() == sum(retained) / len(retained)


@pytest.mark.parametrize("invalid_bit", [-1, 256])
def test_native_unsigned_conversion_refusal_preserves_the_next_sample(invalid_bit: int) -> None:
    """Preserve the unsigned-byte conversion boundary and its nonconsuming failure."""
    native = engine.BitstreamAverager(window=4)
    native.push(1)
    with pytest.raises(OverflowError):
        native.push(invalid_bit)
    assert native.estimate() == 1.0
    native.push(0)
    assert native.estimate() == 0.5


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_native_class_pickle_and_instance_refusal_preserve_continuation(protocol: int) -> None:
    """Round-trip the global class while refusing instance serialization without state loss."""
    assert pickle.loads(pickle.dumps(engine.BitstreamAverager, protocol=protocol)) is (
        engine.BitstreamAverager
    )
    native = engine.BitstreamAverager(window=4)
    native.push(1)
    with pytest.raises(TypeError, match="cannot pickle"):
        pickle.dumps(native, protocol=protocol)
    assert native.estimate() == 1.0
    native.push(0)
    assert native.estimate() == 0.5


def test_native_copy_refusals_preserve_continuation() -> None:
    """Keep the estimator usable after both unsupported instance-copy protocols."""
    native = engine.BitstreamAverager(window=4)
    native.push(1)
    with pytest.raises(TypeError, match="cannot pickle"):
        copy.copy(native)
    with pytest.raises(TypeError, match="cannot pickle"):
        copy.deepcopy(native)
    assert native.estimate() == 1.0
    native.push(0)
    assert native.estimate() == 0.5
