# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/lfsr

module LfsrAccel

using Statistics, LinearAlgebra

mutable struct Lfsr16State
    reg::Float64
end

function Lfsr16State()
    Lfsr16State(0.0)
end

function step(s::Lfsr16State)
    bit = ((s.reg >> 0) ^ (s.reg >> 2)
           ^ (s.reg >> 3) ^ (s.reg >> 5)) & 1
    s.reg = ((s.reg >> 1) | (bit << 15)) & 0xFFFF
    return s.reg
end

function encode(s::Lfsr16State, threshold, bit_length)
    n_words = (bit_length + 31) // 32
    out = [0] * n_words
    for i in 1:bit_length
        val = s.step()
        if val < threshold
            out[i // 32] |= (1 << (i % 32))
    return [w & MASK32 for w in out]
end

function encode_float(s::Lfsr16State, p, bit_length)
    threshold = int(p * 65535)
    return s.encode(threshold, bit_length)
end

end # module LfsrAccel
