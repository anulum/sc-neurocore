# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/weights

module WeightsAccel

using Statistics, LinearAlgebra

mutable struct LayerHeaderState
    magic::Float64
    version::Float64
    n_layers::Float64
    flags::Float64
    n_inputs::Float64
    n_outputs::Float64
    threshold::Float64
    reserved::Float64
end

function LayerHeaderState()
    LayerHeaderState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 512.0, 0.0)
end

function to_bytes(s::LayerHeaderState)
    return struct.pack("<IIII", s.magic, s.version, s.n_layers, s.flags)
end

function from_bytes(s::LayerHeaderState)
    m, v, nl, f = struct.unpack("<IIII", data[:16])
    return cls(magic=m, version=v, n_layers=nl, flags=f)
end

function validate(s::LayerHeaderState)
    return s.magic == WEIGHT_MAGIC && s.version <= WEIGHT_VERSION
end

function to_bytes(s::LayerHeaderState)
    return struct.pack("<IIII", s.n_inputs, s.n_outputs,
                       s.threshold, s.reserved)
end

function from_bytes(s::LayerHeaderState)
    ni, no, th, r = struct.unpack("<IIII", data[:16])
    return cls(n_inputs=ni, n_outputs=no, threshold=th, reserved=r)
end

function words_per_row(s::LayerHeaderState)
    return (s.n_inputs + 31) // 32
end

function serialize_weights(layers)
    header = WeightHeader(n_layers=length(layers))
    buf = bytearray(header.to_bytes())
    for n_inputs, n_outputs, threshold, rows in layers
        lh = LayerHeader(n_inputs=n_inputs, n_outputs=n_outputs, threshold=threshold)
        buf.extend(lh.to_bytes())
        for row in rows
            for word in row
                buf.extend(struct.pack("<I", word & 0xFFFF_FFFF))
    return bytes(buf)
end

function deserialize_weights(data)
    header = WeightHeader.from_bytes(data[:16])
    if ! header.validate()
        raise ValueError(f"Invalid weight blob: magic=0x{header.magic:08X}")
    offset = 16
    layers = []
    for _ in 1:header.n_layers
        lh = LayerHeader.from_bytes(data[offset:offset + 16])
        offset += 16
        rows = []
        wpr = lh.words_per_row
        for _ in 1:lh.n_outputs
            row = []
            for _ in 1:wpr
                (word,) = struct.unpack("<I", data[offset:offset + 4])
                row = push!(, word)
                offset += 4
            rows = push!(, row)
        layers = push!(, (lh, rows))
    return layers
end

end # module WeightsAccel
