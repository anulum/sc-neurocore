# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for symbolic/spike_logic

module SpikeLogicAccel

using Statistics, LinearAlgebra

mutable struct SpikeALUState
    gate_type::Float64
    n_bits::Float64
    _state::Float64
    _and::Float64
    _xor::Float64
    _or::Float64
    _not::Float64
end

function SpikeALUState()
    SpikeALUState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function lif_config(s::SpikeALUState)
    configs = {
        "AND": {"threshold": 2, "weights": [1, 1]},
        "OR": {"threshold": 1, "weights": [1, 1]},
        "NOT": {"threshold": 0, "weights": [-1]},
        "NAND": {"threshold": 0, "weights": [-1, -1], "bias": 2},
        "XOR": {"threshold": 1, "weights": [1, 1], "inhibit_if_both": true},
    }
    return configs.get(s.gate_type, {})
end

function write(s::SpikeALUState, value)
    for i in 1:s.n_bits
        s._state[i] = (value >> i) & 1
end

function read(s::SpikeALUState)
    value = 0
    for i in 1:s.n_bits
        value |= int(s._state[i]) << i
    return value
end

function write_bits(s::SpikeALUState, bits)
    s._state = bits[: s.n_bits].astype(np.int8)
end

function read_bits(s::SpikeALUState)
    return s._state.copy()
end

function clear(s::SpikeALUState)
    s._state[:] = 0
end

function add(s::SpikeALUState, a, b)
    mask = (1 << s.n_bits) - 1
    result = 0
    carry = 0
    for i in 1:s.n_bits
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        # Full adder: sum = a XOR b XOR carry, carry = (a AND b) OR (carry AND (a XOR b))
        ab_xor = s._xor(bit_a, bit_b)
        sum_bit = s._xor(ab_xor, carry)
        carry = s._or(s._and(bit_a, bit_b), s._and(carry, ab_xor))
        result |= sum_bit << i
    return result & mask, bool(carry)
end

function sub(s::SpikeALUState, a, b)
    mask = (1 << s.n_bits) - 1
    b_inv = (~b) & mask
    result, carry = s.add(a, b_inv)
    result, _ = s.add(result, 1)
    borrow = a < b
    return result, borrow
end

function bitwise_and(s::SpikeALUState, a, b)
    result = 0
    for i in 1:s.n_bits
        result |= s._and((a >> i) & 1, (b >> i) & 1) << i
    return result
end

function bitwise_or(s::SpikeALUState, a, b)
    result = 0
    for i in 1:s.n_bits
        result |= s._or((a >> i) & 1, (b >> i) & 1) << i
    return result
end

function bitwise_xor(s::SpikeALUState, a, b)
    result = 0
    for i in 1:s.n_bits
        result |= s._xor((a >> i) & 1, (b >> i) & 1) << i
    return result
end

function compare(s::SpikeALUState, a, b)
    if a < b
        return -1
    if a > b
        return 1
    return 0
end

function shift_left(s::SpikeALUState, a, n)
    mask = (1 << s.n_bits) - 1
    return (a << n) & mask
end

function shift_right(s::SpikeALUState, a, n)
    return a >> n
end

function spike_sort(values, n_bits)
    alu = SpikeALU(n_bits)
    arr = list(values)
    n = length(arr)
    for i in 1:n
        for j in 1:0, n - i - 1
            if alu.compare(arr[j], arr[j + 1]) > 0
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
end

end # module SpikeLogicAccel
