# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo deterministic complete dense IF replay

"""Native dense-if-f64-sequential-v1 replay; compile with FP contraction disabled."""

from ann_to_snn_parameters import DenseLayer, ConvertedSNN
from ann_to_snn_resources import admit_replay, checked_add
from std.math import isfinite
from ann_to_snn_compute import ReplayResult, replay_owned


def replay(model: ConvertedSNN, frames: List[Float64], steps: Int, batch: Int, initial_state: List[List[Float64]] = List[List[Float64]](), use_initial_state: Bool = False, trace: Bool = False, binary_inputs: Bool = True, max_working_bytes: Int = 268435456) raises -> ReplayResult:
    """Replay row-major explicit frames with inclusive IF events and subtractive reset.

    Bias applies every step after ascending column multiply/add reductions. Layers
    consume same-step events. Linear output retains cumulative signed current;
    IF output counts incremental spikes. Empty axes preserve supplied states.
    State/event traces capture every post-reset timestep. Numeric admission uses
    8*(2P+2F+2S+2O+H+5M+I), excluding caller and runtime overhead. Refusal raises
    Error with IF invalid input, IF resource limit or IF overflow and does not edit
    caller state. Explicit use_initial_state distinguishes empty supplied state
    from the default preloads. Returned arrays are independently owned.

    Args:
        model: Dense stack; public parameters are snapshotted and revalidated.
        frames: Flat time/batch/input finite currents in the unit interval.
        steps: Nonnegative number of explicit input timesteps.
        batch: Nonnegative number of samples.
        initial_state: Batch/output state per layer when explicitly selected.
        use_initial_state: Select supplied state instead of per-layer preloads.
        trace: Retain complete state and IF event trajectories.
        binary_inputs: Require exact zero/one events when true.
        max_working_bytes: Positive addressable numeric allocation limit.

    Returns:
        Independently owned output, final states and requested full traces.

    Raises:
        Error: Invalid input, numeric reservation refusal or finite arithmetic overflow.
    """
    if steps < 0 or batch < 0 or len(model.layers) == 0:
        raise Error("IF invalid input")
    var snapshot = ConvertedSNN(model.layers, model.linear, max_working_bytes)
    var input_width = snapshot.layers[0].inputs
    if steps != 0 and batch > 0x7FFFFFFFFFFFFFFF // steps:
        raise Error("IF invalid input")
    var extent = steps * batch
    if extent != 0 and input_width > 0x7FFFFFFFFFFFFFFF // extent:
        raise Error("IF invalid input")
    if extent * input_width != len(frames):
        raise Error("IF invalid input")
    if use_initial_state and len(initial_state) != len(snapshot.layers):
        raise Error("IF invalid input")
    var coefficients = 0
    var widths = List[Int]()
    for i in range(len(snapshot.layers)):
        coefficients = checked_add(coefficients, checked_add(len(snapshot.layers[i].weights), len(snapshot.layers[i].bias)))
        widths.append(snapshot.layers[i].outputs)
    admit_replay(coefficients, input_width, widths, steps, batch, trace, snapshot.linear, max_working_bytes)
    var owned_frames = List[Float64](capacity=len(frames))
    for value in frames:
        owned_frames.append(value)
    var owned_states = List[List[Float64]]()
    if use_initial_state:
        for index in range(len(initial_state)):
            if len(initial_state[index]) != batch * snapshot.layers[index].outputs:
                raise Error("IF invalid input")
            for value in initial_state[index]:
                if not isfinite(value):
                    raise Error("IF invalid input")
        for index in range(len(initial_state)-1, -1, -1):
            var state = List[Float64](capacity=len(initial_state[index]))
            for value in initial_state[index]:
                state.append(value)
            owned_states.append(state^)
    return replay_owned(snapshot^, owned_frames^, steps, batch, owned_states^, use_initial_state, trace, binary_inputs)


def classify(model: ConvertedSNN, result: ReplayResult, batch: Int) raises -> List[Int]:
    """Select first maximal finite output in each row, returning zero-based labels.

    Args:
        model: Stack declaring the final output width.
        result: Replay response with finite batch/output values.
        batch: Nonnegative number of classification rows.

    Returns:
        First-maximum zero-based labels, one per sample.

    Raises:
        Error: Invalid geometry, values or addressable classification size.
    """
    if batch < 0 or len(model.layers) == 0:
        raise Error("IF invalid input")
    var width = model.layers[len(model.layers)-1].outputs
    if width <= 0:
        raise Error("IF invalid input")
    if batch != 0 and width > 0x7FFFFFFFFFFFFFFF // batch:
        raise Error("IF invalid input")
    if batch * width != len(result.output):
        raise Error("IF invalid input")
    for value in result.output:
        if not isfinite(value):
            raise Error("IF invalid input")
    var labels = List[Int]()
    for row in range(batch):
        var best = 0
        for node in range(1, width):
            if result.output[row*width+node] > result.output[row*width+best]:
                best = node
        labels.append(best)
    return labels^
