# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo consuming dense IF computation

"""Native dense-if-f64-sequential-v1 replay; compile with FP contraction disabled."""

from std.math import isfinite
from ann_to_snn_parameters import ConvertedSNN


struct ReplayResult(Copyable, Movable):
    """Owned output, final states and layer-major time/batch/output traces."""
    var output: List[Float64]
    """Batch-major incremental counts or cumulative signed linear current."""
    var final_state: List[List[Float64]]
    """Owned batch/output state for each weighted layer."""
    var state_trace: List[List[Float64]]
    """Owned time/batch/output post-reset states for each layer."""
    var spike_trace: List[List[Float64]]
    """Owned time/batch/output events for IF layers only."""

    def __init__(out self):
        """Construct empty result containers; each replay fills independently owned storage."""
        self.output = List[Float64]()
        self.final_state = List[List[Float64]]()
        self.state_trace = List[List[Float64]]()
        self.spike_trace = List[List[Float64]]()



def replay_owned(var snapshot: ConvertedSNN, var owned_frames: List[Float64], steps: Int, batch: Int, var initial_state: List[List[Float64]], use_initial_state: Bool, trace: Bool, binary_inputs: Bool) raises -> ReplayResult:
    """Consume independently owned, metadata-admitted buffers using ordered float64 IF dynamics.

    Args:
        snapshot: Validated owned coefficient stack, transferred without another copy.
        owned_frames: Admitted row-major frames, transferred without another copy.
        steps: Admitted nonnegative timestep count.
        batch: Admitted nonnegative sample count.
        initial_state: Owned states in reverse layer order for move-only removal.
        use_initial_state: Select supplied states instead of finite preloads.
        trace: Retain complete independently owned state/event traces.
        binary_inputs: Require exact zero/one drive, otherwise finite unit currents.

    Returns:
        Owned full result; numeric buffers are not borrowed from transferred inputs.

    Raises:
        Error: Invalid drive/state domains or finite arithmetic overflow.
    """
    var input_width = snapshot.layers[0].inputs
    for value in owned_frames:
        if not isfinite(value) or value < 0 or value > 1 or (binary_inputs and value != 0 and value != 1):
            raise Error("IF invalid input")
    var spiking = len(snapshot.layers) - Int(snapshot.linear)
    var result = ReplayResult()
    for index in range(len(snapshot.layers)):
        var width = batch * snapshot.layers[index].outputs
        var state = List[Float64](capacity=width)
        if use_initial_state:
            state = initial_state.pop()
            if len(state) != width:
                raise Error("IF invalid input")
            for value in state:
                if not isfinite(value):
                    raise Error("IF invalid input")
        else:
            var shift = Float64(0)
            if index < spiking:
                shift = snapshot.layers[index].initial_fraction * snapshot.layers[index].threshold
            if not isfinite(shift):
                raise Error("IF overflow")
            for _ in range(width):
                state.append(shift)
        result.final_state.append(state^)
        if trace:
            var states = List[Float64](capacity=steps * width)
            var spikes = List[Float64](capacity=steps * width if index < spiking else 0)
            for _ in range(steps * width):
                states.append(0)
                if index < spiking:
                    spikes.append(0)
            result.state_trace.append(states^)
            if index < spiking:
                result.spike_trace.append(spikes^)
    result.output = List[Float64](capacity=batch * snapshot.layers[len(snapshot.layers)-1].outputs)
    for _ in range(batch * snapshot.layers[len(snapshot.layers)-1].outputs):
        result.output.append(0)
    var active_steps = steps
    if batch == 0:
        active_steps = 0
    for step in range(active_steps):
        var drive = List[Float64](capacity=batch * input_width)
        for i in range(batch * input_width):
            drive.append(owned_frames[step*batch*input_width+i])
        for index in range(len(snapshot.layers)):
            var events = List[Float64](capacity=batch * snapshot.layers[index].outputs)
            for _ in range(batch * snapshot.layers[index].outputs):
                events.append(0)
            for row in range(batch):
                for node in range(snapshot.layers[index].outputs):
                    var current = Float64(0)
                    for column in range(snapshot.layers[index].inputs):
                        var product = drive[row*snapshot.layers[index].inputs+column] * snapshot.layers[index].weights[node*snapshot.layers[index].inputs+column]
                        current = current + product
                    if len(snapshot.layers[index].bias) != 0:
                        current = current + snapshot.layers[index].bias[node]
                    var slot = row*snapshot.layers[index].outputs+node
                    var state = result.final_state[index][slot] + current
                    if not isfinite(current) or not isfinite(state):
                        raise Error("IF overflow")
                    if index < spiking:
                        var event = Float64(state >= snapshot.layers[index].threshold)
                        state = state - event * snapshot.layers[index].threshold
                        events[slot] = event
                        if trace:
                            result.spike_trace[index][step*batch*snapshot.layers[index].outputs+slot] = event
                        if index == len(snapshot.layers)-1:
                            result.output[slot] += event
                    result.final_state[index][slot] = state
                    if trace:
                        result.state_trace[index][step*batch*snapshot.layers[index].outputs+slot] = state
            drive = events^
    if snapshot.linear:
        result.output = result.final_state[len(snapshot.layers)-1].copy()
    return result^
