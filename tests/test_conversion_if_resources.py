# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real public dense IF buffer admission

"""Admit exact numeric budgets and refuse virtual huge arrays before copies."""

from typing import Literal

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, 2**64])
def test_invalid_operator_budget_refused(budget: int) -> None:
    """Require a positive addressable integer even for a tiny real replay."""
    with pytest.raises(ValueError, match="working byte budget"):
        ConvertedSNN([[[1.0]]], [None], [1.0], T=1, max_working_bytes=budget)


@pytest.mark.parametrize(
    "mode,trace,budget", [("spikes", False, 112), ("spikes", True, 128), ("linear", True, 120)]
)
def test_exact_replay_reservation_boundary(mode: OutputMode, trace: bool, budget: int) -> None:
    """Include real traces and accept exactly the documented reservation."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1, output_mode=mode)
    frames = np.ones((1, 1, 1))
    initial = np.zeros((1, 1))
    with pytest.raises(MemoryError, match="replay buffers"):
        model.replay(frames, initial_state=[initial], trace=trace, max_working_bytes=budget - 1)
    result = model.replay(frames, initial_state=[initial], trace=trace, max_working_bytes=budget)
    np.testing.assert_array_equal(result.output, [[1.0]])
    np.testing.assert_array_equal(initial, [[0.0]])


@pytest.mark.parametrize("entry", ["run", "rates", "classify"])
def test_encoded_entrypoints_propagate_operator_budget(
    entry: Literal["run", "rates", "classify"],
) -> None:
    """The public encoder and decoders share the same strict numeric reservation."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    call = model.run if entry == "run" else model.rates if entry == "rates" else model.classify
    with pytest.raises(MemoryError, match="replay buffers"):
        call([1.0], max_working_bytes=111)
    actual = call([1.0], max_working_bytes=112)
    np.testing.assert_array_equal(actual, 0 if entry == "classify" else [1.0])


def test_virtual_huge_parameter_refused_before_snapshot() -> None:
    """Refuse a broadcast matrix without materializing its billion coefficients."""
    weights = np.broadcast_to(np.array([[1.0]]), (2**30, 1))
    with pytest.raises(MemoryError, match="coefficient snapshots"):
        ConvertedSNN([weights], [None], [1.0], T=1)


def test_virtual_huge_frames_refused_before_copy() -> None:
    """Refuse frame metadata without copying the large broadcast input."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    frames = np.broadcast_to(np.array([[[1.0]]]), (2**30, 1, 1))
    with pytest.raises(MemoryError, match="replay buffers"):
        model.replay(frames)


def test_zero_time_huge_batch_still_requires_state_budget() -> None:
    """A zero-byte input cannot authorize enormous nonempty membrane arrays."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    frames = np.empty((0, 2**30, 1))
    assert frames.nbytes == 0
    with pytest.raises(MemoryError, match="replay buffers"):
        model.replay(frames, trace=True)


def test_virtual_huge_initial_shape_refused_before_copy() -> None:
    """Check state geometry before materializing a wrong-size broadcast state."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    initial = np.broadcast_to(np.array([[0.0]]), (2**30, 1))
    with pytest.raises(ValueError, match="initial state dimensions"):
        model.replay(np.zeros((1, 1, 1)), initial_state=[initial])


def test_zero_batch_maximal_time_does_not_iterate() -> None:
    """Empty public input returns immediately at the largest exact count budget."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=2**53)
    result = model.replay(np.empty((2**53, 0, 1)), trace=True, max_working_bytes=16)
    assert result.output.shape == (0, 1)
    assert result.state_trace[0].shape == (2**53, 0, 1)
    assert model.run(np.empty((0, 1)), max_working_bytes=16).shape == (0, 1)


@pytest.mark.parametrize("encoding", ["poisson", "constant"])
def test_virtual_huge_encoder_input_refused_before_copy(
    encoding: Literal["poisson", "constant"],
) -> None:
    """The public encoder checks batch metadata before float64 or RNG buffers."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    values = np.broadcast_to(np.array([[1.0]]), (2**30, 1))
    with pytest.raises(MemoryError, match="replay buffers"):
        model.run(values, input_mode=encoding)
