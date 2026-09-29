# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia complete native request admission

"""Admit raw C descriptors and replay one canonical borrowed-parameter snapshot.

Raw pointers require complete live aligned arrays unchanged during the call.
Address checks cannot prove OS accessibility. Positive addressable dimensions,
reserved flags, coefficient geometry and complete copy budgets are checked before
numeric snapshots. The canonical replay owns every returned state/event vector.
"""
function execute(request::ReplayRequest)
    maximum = UInt(typemax(Int))
    request.version == 1 && request.flags & ~UInt32(15) == 0 && request.layer_count > 0 &&
        0 < request.max_working_bytes <= maximum && request.steps <= maximum && request.batch <= maximum ||
        throw(ArgumentError("invalid native replay request"))
    request.layer_count <= request.max_working_bytes ÷ 16 || throw(OutOfMemoryError())
    specs = borrowed(request.layers, request.layer_count)
    coefficients = BigInt(0)
    for layer in specs
        layer.inputs > 0 && layer.outputs > 0 && BigInt(layer.inputs) * layer.outputs <= maximum &&
            (layer.bias_len == 0 || layer.bias_len == layer.outputs) ||
            throw(ArgumentError("invalid native coefficient geometry"))
        coefficients += BigInt(layer.inputs) * layer.outputs + layer.bias_len
    end
    16coefficients <= request.max_working_bytes || throw(OutOfMemoryError())
    parameters = AnnToSnnAccel.LayerParameters[]
    for layer in specs
        weight = borrowed(layer.weights, layer.inputs * layer.outputs)
        bias = layer.bias_len == 0 ? nothing : borrowed(layer.bias, layer.bias_len)
        push!(parameters, AnnToSnnAccel.LayerParameters(Int(layer.inputs), Int(layer.outputs),
              weight, bias, layer.threshold, layer.initial_fraction))
    end
    frames = borrowed(request.frames, request.frames_len)
    initial = request.flags & 8 == 0 ? nothing : [borrowed(l.initial, l.initial_len) for l in specs]
    return AnnToSnnAccel.replay_parameters(parameters, frames, (Int(request.steps), Int(request.batch));
        output_mode=request.flags & 4 == 0 ? :spikes : :linear, initial_state=initial,
        trace=request.flags & 1 != 0, binary_inputs=request.flags & 2 != 0,
        max_working_bytes=Int(request.max_working_bytes))
end
