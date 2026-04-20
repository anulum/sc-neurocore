# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/bitstream

module BitstreamAccel

using Statistics, LinearAlgebra

function popcount32(word)
    x = word & MASK32
    x = x - ((x >> 1) & 0x5555_5555)
    x = (x & 0x3333_3333) + ((x >> 2) & 0x3333_3333)
    x = (x + (x >> 4)) & 0x0F0F_0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F
end

function popcount_slice(words)
    total = 0
    for w in words
        total += popcount32(w)
    return total
end

function sc_and(a, b)
    return (a & b) & MASK32
end

function sc_or(a, b)
    return (a | b) & MASK32
end

function sc_xor(a, b)
    return (a ^ b) & MASK32
end

function sc_sub(a, b)
    return (a & (~b & MASK32)) & MASK32
end

function sc_mux(a, b, sel)
    return ((a & sel) | (b & (~sel & MASK32))) & MASK32
end

function and_packed(a, b)
    assert length(a) == length(b)
    return [(x & y) & MASK32 for x, y in zip(a, b)]
end

function mux_packed(a, b, sel)
    assert length(a) == length(b) == length(sel)
    return [((x & s) | (y & (~s & MASK32))) & MASK32
            for x, y, s in zip(a, b, sel)]
end

function probability(words, bit_length)
    if bit_length == 0
        return 0.0
    return popcount_slice(words) / bit_length
end

function scc(a, b, bit_length)
    assert length(a) == length(b)
    if bit_length == 0
        return 0.0
    n = float(bit_length)
    pa = popcount_slice(a) / n
    pb = popcount_slice(b) / n
    and_count = sum(popcount32(x & y) for x, y in zip(a, b))
    p_and = and_count / n
    num = p_and - (pa * pb)
    if abs(num) < 1e-7
        return 0.0
    if num > 0.0
        denom = min(pa, pb) - (pa * pb)
    else
        denom = (pa * pb) - max(pa + pb - 1.0, 0.0)
    if abs(denom) < 1e-7
        return 0.0
    return max(-1.0, min(1.0, num / denom))
end

end # module BitstreamAccel
