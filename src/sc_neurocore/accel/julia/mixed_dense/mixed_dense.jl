# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mixed-precision Q8.8×Q16.16 dense MAC (batch)

"""
Bit-exact Julia port of the integer mixed-precision dense MAC in
`src/sc_neurocore/compiler/mixed_dense_kernel.py` and
`engine/src/ir/qformat.rs`.

Q8.8 weights contract Q16.16 input codes in an Int64 accumulator (the caller
keeps the contraction within Int64 range); the accumulator divides by the Q8.8
weight scale with an arithmetic right shift (floor division for the power-of-two
scale) and saturates to the Q16.16 code range. The arithmetic is exact integer,
so this backend matches the Python, Rust, Go and Mojo references bit-for-bit.
"""
module MixedDenseAccel

export mixed_dense_forward_batch_q88_q1616!

const WEIGHT_FRACTION = 8
const I32_MAX = Int64(2147483647)
const I32_MIN = Int64(-2147483648)

"""Batched mixed-precision dense MAC; fills the pre-allocated buffers in place."""
function mixed_dense_forward_batch_q88_q1616!(
    weights::AbstractVector{<:Integer},
    inputs::AbstractVector{<:Integer},
    n_outputs::Integer,
    n_inputs::Integer,
    outputs::AbstractVector{<:Integer},
    overflow::AbstractVector{<:Integer},
    underflow::AbstractVector{<:Integer},
)
    outs = Int64(n_outputs)
    ins = Int64(n_inputs)
    n_batch = div(Int64(length(inputs)), ins)
    @inbounds for b in 0:(n_batch - 1)
        for o in 0:(outs - 1)
            sum = Int64(0)
            for i in 0:(ins - 1)
                sum += Int64(weights[o * ins + i + 1]) * Int64(inputs[b * ins + i + 1])
            end
            scaled = sum >> WEIGHT_FRACTION
            idx = b * outs + o + 1
            if scaled > I32_MAX
                outputs[idx] = I32_MAX
                overflow[idx] = 1
                underflow[idx] = 0
            elseif scaled < I32_MIN
                outputs[idx] = I32_MIN
                overflow[idx] = 1
                underflow[idx] = 0
            else
                outputs[idx] = scaled
                overflow[idx] = 0
                underflow[idx] = (sum != 0 && scaled == 0) ? 1 : 0
            end
        end
    end
    return nothing
end

end # module MixedDenseAccel
