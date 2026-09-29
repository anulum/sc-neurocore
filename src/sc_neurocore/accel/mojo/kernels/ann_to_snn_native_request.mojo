# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo dense IF C request admission

"""Admit every extent and numeric reservation before copying caller-owned doubles."""

from std.memory import Pointer
from ann_to_snn_native_types import ReplayRequest, LayerSpec, checked_count, check_span, copy_doubles
from ann_to_snn_parameters import DenseLayer, ConvertedSNN
from ann_to_snn_resources import admit_copy, admit_replay, checked_add, checked_mul
from ann_to_snn_compute import ReplayResult, replay_owned


def execute(request: ReplayRequest) raises -> ReplayResult:
    """Snapshot admitted C storage once and consume it through the canonical IF kernel.

    Args:
        request: Live immutable ABI-one descriptor; all arrays remain stable during the call.

    Returns:
        Independent result owner with complete requested trajectories.

    Raises:
        Error: Invalid descriptors/domains, resource refusal or finite arithmetic overflow.
    """
    if request.version != 1 or request.flags & UInt32(0xFFFFFFF0) != 0:
        raise Error("IF invalid input")
    var count = checked_count(request.layer_count)
    var steps = checked_count(request.steps)
    var batch = checked_count(request.batch)
    var maximum = checked_count(request.max_working_bytes)
    var frames_len = checked_count(request.frames_len)
    if count == 0 or maximum == 0:
        raise Error("IF invalid input")
    if count > maximum // 16:
        raise Error("IF resource limit")
    check_span(request.layers, count, 72)
    var source = Pointer[LayerSpec, ImmUntrackedOrigin](unsafe_from_address=Int(request.layers))
    var specs = List[LayerSpec](capacity=count)
    var widths = List[Int](capacity=count)
    var coefficients = 0
    var previous = 0
    for i in range(count):
        var layer = source[unsafe_offset=i].copy()
        var inputs = checked_count(layer.inputs)
        var outputs = checked_count(layer.outputs)
        var bias_len = checked_count(layer.bias_len)
        if inputs == 0 or outputs == 0 or inputs > 0x7FFFFFFFFFFFFFFF // outputs:
            raise Error("IF invalid input")
        if (i > 0 and inputs != previous) or (bias_len != 0 and bias_len != outputs):
            raise Error("IF invalid input")
        check_span(layer.weights, inputs * outputs, 8)
        check_span(layer.bias, bias_len, 8)
        coefficients = checked_add(coefficients, checked_add(inputs * outputs, bias_len))
        widths.append(outputs)
        specs.append(layer^)
        previous = outputs
    admit_copy(coefficients, maximum)
    var inputs = Int(specs[0].inputs)
    if steps != 0 and batch > 0x7FFFFFFFFFFFFFFF // steps:
        raise Error("IF invalid input")
    var extent = steps * batch
    if extent != 0 and inputs > 0x7FFFFFFFFFFFFFFF // extent:
        raise Error("IF invalid input")
    if frames_len != extent * inputs:
        raise Error("IF invalid input")
    check_span(request.frames, frames_len, 8)
    var trace = request.flags & 1 != 0
    var linear = request.flags & 4 != 0
    var use_initial = request.flags & 8 != 0
    admit_replay(coefficients, inputs, widths, steps, batch, trace, linear, maximum)
    if use_initial:
        for i in range(count):
            var state_len = checked_count(specs[i].initial_len)
            if state_len != checked_mul(batch, widths[i]):
                raise Error("IF invalid input")
            check_span(specs[i].initial, state_len, 8)
    var layers = List[DenseLayer](capacity=count)
    for i in range(count):
        var weight = copy_doubles(specs[i].weights, Int(specs[i].inputs) * widths[i])
        var bias = copy_doubles(specs[i].bias, Int(specs[i].bias_len))
        layers.append(DenseLayer(Int(specs[i].inputs), widths[i], owned_weights=weight^, owned_bias=bias^, threshold=specs[i].threshold, initial_fraction=specs[i].initial_fraction, max_working_bytes=maximum))
    var model = ConvertedSNN(owned_layers=layers^, linear=linear, max_working_bytes=maximum)
    var frames = copy_doubles(request.frames, frames_len)
    var initial = List[List[Float64]](capacity=count if use_initial else 0)
    if use_initial:
        for i in range(count-1, -1, -1):
            initial.append(copy_doubles(specs[i].initial, Int(specs[i].initial_len)))
    return replay_owned(model^, frames^, steps, batch, initial^, use_initial, trace, request.flags & 2 != 0)
