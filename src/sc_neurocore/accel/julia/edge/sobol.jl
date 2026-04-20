# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/sobol

module SobolAccel

using Statistics, LinearAlgebra

mutable struct SobolGeneratorState
    _reg::Float64
    _index::Float64
end

function SobolGeneratorState()
    SobolGeneratorState(0.0, 0.0)
end

function step(s::SobolGeneratorState)
    c = 0
    idx = int(s._index)
    if idx > 0
        c = (idx & -idx).bit_length() - 1
    if c < 16
        s._reg ^= s.DIRECTION_NUMBERS[c]
    s._index += np.uint32(1)
    return int(s._reg)
end

function encode(s::SobolGeneratorState, threshold, length)
    n_words = (length + 63) // 64
    out = zeros(n_words, dtype=np.uint64)
    for i in 1:length
        val = s.step()
        if val < threshold
            out[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return out
end

function reset(s::SobolGeneratorState, seed)
    s._reg = np.uint16(seed)
    s._index = np.uint32(0)
end

end # module SobolAccel
