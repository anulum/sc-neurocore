# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public dense IF replay state and event fidelity

"""Compare actual public IF replay against analytical and independent recurrences."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import FloatArray, OutputMode


def _reference(
    weights: Sequence[FloatArray],
    biases: Sequence[FloatArray | None],
    thresholds: Sequence[float],
    frames: FloatArray,
    shift: float,
    linear: bool,
) -> tuple[FloatArray, list[FloatArray], list[FloatArray]]:
    """Evaluate the declared recurrence independently with scalar Python arithmetic."""
    steps, batch, _ = frames.shape
    state = [
        [
            [
                0.0 if linear and i == len(weights) - 1 else shift * thresholds[i]
                for _ in range(w.shape[0])
            ]
            for _ in range(batch)
        ]
        for i, w in enumerate(weights)
    ]
    states = [np.empty((steps, batch, w.shape[0])) for w in weights]
    events = [
        np.empty((steps, batch, w.shape[0]))
        for i, w in enumerate(weights)
        if not (linear and i == len(weights) - 1)
    ]
    output = np.zeros((batch, weights[-1].shape[0]))
    for step in range(steps):
        for row in range(batch):
            drive = [float(v) for v in frames[step, row]]
            for layer, (w, b, threshold) in enumerate(zip(weights, biases, thresholds)):
                next_drive: list[float] = []
                for node in range(w.shape[0]):
                    current = 0.0
                    for column in range(w.shape[1]):
                        current += drive[column] * float(w[node, column])
                    if b is not None:
                        current += float(b[node])
                    state[layer][row][node] += current
                    if not (linear and layer == len(weights) - 1):
                        spike = float(state[layer][row][node] >= threshold)
                        state[layer][row][node] -= spike * threshold
                        events[layer][step, row, node] = spike
                        next_drive.append(spike)
                        if layer == len(weights) - 1:
                            output[row, node] += spike
                    states[layer][step, row, node] = state[layer][row][node]
                drive = next_drive
    if linear:
        output = np.array(state[-1], dtype=np.float64)
    return output, states, events


def test_analytical_bias_shift_and_subtractive_reset_trace() -> None:
    """Verify every state and event of a known constant-current IF trajectory."""
    model = ConvertedSNN([[[1.0]]], [[0.25]], [1.0], T=6, initial_membrane_fraction=0.5)
    frames = np.array([0.0, 0.25, 0.75, 1.0, 0.0, 0.5]).reshape(6, 1, 1)
    result = model.replay(frames, binary_inputs=False, trace=True)
    np.testing.assert_array_equal(result.output, [[4.0]])
    np.testing.assert_array_equal(result.state_trace[0].ravel(), [0.75, 0.25, 0.25, 0.5, 0.75, 0.5])
    np.testing.assert_array_equal(result.spike_trace[0].ravel(), [0, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(result.final_state[0], [[0.5]])


def test_threshold_equality_and_same_step_layer_propagation() -> None:
    """A threshold-equality event reaches the next IF within the same timestep."""
    model = ConvertedSNN([[[1.0]], [[1.0]]], [None, None], [1.0, 1.0], T=3)
    result = model.replay(np.array([1.0, 0.0, 1.0]).reshape(3, 1, 1), trace=True)
    np.testing.assert_array_equal(result.output, [[2.0]])
    for states, spikes in zip(result.state_trace, result.spike_trace):
        np.testing.assert_array_equal(states.ravel(), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(spikes.ravel(), [1.0, 0.0, 1.0])


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("binary", [False, True])
def test_complete_states_match_independent_scalar_oracle(mode: OutputMode, binary: bool) -> None:
    """Compare non-dyadic signed coefficients without sharing implementation arithmetic."""
    rng = np.random.default_rng(1397)
    weights = [rng.normal(0, 0.8, (5, 3)), rng.normal(0, 0.6, (2, 5))]
    biases = [rng.normal(0, 0.1, 5), rng.normal(0, 0.1, 2)]
    frames = rng.random((17, 3, 3))
    if binary:
        frames = (frames < 0.5).astype(np.float64)
    model = ConvertedSNN(
        weights, biases, [1.25, 0.75], T=17, initial_membrane_fraction=0.5, output_mode=mode
    )
    result = model.replay(frames, binary_inputs=binary, trace=True)
    expected, states, spikes = _reference(
        weights, biases, [1.25, 0.75], frames, 0.5, mode == "linear"
    )
    np.testing.assert_array_equal(result.output, expected)
    for actual, correct in zip(result.state_trace, states):
        np.testing.assert_array_equal(actual, correct)
    for actual, correct in zip(result.spike_trace, spikes):
        np.testing.assert_array_equal(actual, correct)
    for final, correct in zip(result.final_state, states):
        np.testing.assert_array_equal(final, correct[-1])


@pytest.mark.parametrize("mode", ["spikes", "linear"])
def test_chunked_state_and_trace_equal_full_replay(mode: OutputMode) -> None:
    """Continue a real replay without resetting membranes or the readout integrator."""
    rng = np.random.default_rng(177)
    model = ConvertedSNN(
        [rng.normal(size=(4, 3)), rng.normal(size=(2, 4))],
        [None, np.array([0.25, -0.5])],
        [1.0, 1.5],
        T=13,
        output_mode=mode,
    )
    frames = (rng.random((13, 2, 3)) < 0.5).astype(np.float64)
    full = model.replay(frames, trace=True)
    first = model.replay(frames[:5], trace=True)
    initial = tuple(values.copy() for values in first.final_state)
    second = model.replay(frames[5:], initial_state=initial, trace=True)
    expected_output = second.output if mode == "linear" else first.output + second.output
    np.testing.assert_array_equal(expected_output, full.output)
    for before, after, complete, original, initial_value in zip(
        first.state_trace,
        second.state_trace,
        full.state_trace,
        first.final_state,
        initial,
    ):
        np.testing.assert_array_equal(np.concatenate([before, after]), complete)
        np.testing.assert_array_equal(original, initial_value)
    for actual, correct in zip(second.final_state, full.final_state):
        np.testing.assert_array_equal(actual, correct)
    for before, after, correct in zip(first.spike_trace, second.spike_trace, full.spike_trace):
        np.testing.assert_array_equal(np.concatenate([before, after]), correct)


def test_arbitrary_initial_state_emits_only_one_event_per_step() -> None:
    """Drain a superthreshold initial membrane using the declared single-event reset."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=3)
    initial = np.array([[2.5]])
    result = model.replay(np.zeros((3, 1, 1)), initial_state=[initial], trace=True)
    np.testing.assert_array_equal(result.spike_trace[0].ravel(), [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.state_trace[0].ravel(), [1.5, 0.5, 0.5])
    np.testing.assert_array_equal(initial, [[2.5]])


@pytest.mark.parametrize("mode", ["spikes", "linear"])
def test_empty_time_preserves_owned_initial_state(mode: OutputMode) -> None:
    """An empty replay is a state identity, including an existing signed readout."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1, output_mode=mode)
    initial = np.array([[-0.75]])
    result = model.replay(np.empty((0, 1, 1)), initial_state=[initial], trace=True)
    np.testing.assert_array_equal(result.final_state[0], initial)
    np.testing.assert_array_equal(result.output, initial if mode == "linear" else [[0.0]])
    result.final_state[0][0, 0] = 100
    np.testing.assert_array_equal(initial, [[-0.75]])


def test_empty_batch_and_trace_disabled() -> None:
    """Retain valid zero-row geometry without fabricating observations."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=2)
    result = model.replay(np.empty((2, 0, 1)))
    assert result.output.shape == (0, 1) and result.final_state[0].shape == (0, 1)
    assert result.state_trace == () and result.spike_trace == ()


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((2, 1)),
        np.zeros((2, 1, 2)),
        np.array([[[1.5]]]),
        np.array([[[float("nan")]]]),
        np.array([[[0.5]]]),
    ],
)
def test_invalid_frames_refused_without_mutating_initial_state(frames: FloatArray) -> None:
    """Refuse shape, nonfinite, out-of-range and nonbinary data before evolution."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    initial = np.array([[0.5]])
    with pytest.raises(ValueError):
        model.replay(frames, initial_state=[initial])
    np.testing.assert_array_equal(initial, [[0.5]])


@pytest.mark.parametrize("initial", [[], [np.zeros((2, 1))], [np.array([[float("inf")]])]])
def test_invalid_initial_state_refused(initial: Sequence[npt.ArrayLike]) -> None:
    """Require one finite correctly shaped state for every weighted layer."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=1)
    with pytest.raises(ValueError):
        model.replay(np.ones((1, 1, 1)), initial_state=initial)


def test_finite_drive_overflow_is_atomic_for_caller_state() -> None:
    """Actual finite arithmetic overflow cannot publish a partial updated state."""
    model = ConvertedSNN([[[1e308]]], [[1e308]], [1.0], T=2)
    initial = np.array([[0.25]])
    with pytest.raises(FloatingPointError):
        model.replay(np.ones((2, 1, 1)), initial_state=[initial], trace=True)
    np.testing.assert_array_equal(initial, [[0.25]])


def test_default_membrane_overflow_refused() -> None:
    """Detect a nonrepresentable default preload before producing a replay result."""
    model = ConvertedSNN([[[1.0]]], [None], [1e308], T=1, initial_membrane_fraction=1e308)
    with pytest.raises(FloatingPointError, match="initial membrane overflow"):
        model.replay(np.zeros((1, 1, 1)))
