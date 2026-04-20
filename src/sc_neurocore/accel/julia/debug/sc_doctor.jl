# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for debug/sc_doctor

module ScDoctorAccel

using Statistics, LinearAlgebra

mutable struct ScDoctorState
    current_bitstream_length::Float64
    target_precision::Float64
    error_correction_enabled::Float64
end

function ScDoctorState()
    ScDoctorState(0.0, 0.0, 0.0)
end

function adapt(s::ScDoctorState, current_correlation, popcount)
    if current_correlation > 0.15
        s.current_bitstream_length *= 2
        if s.current_bitstream_length > 2048
            s.error_correction_enabled = true
    elseif current_correlation < 0.05 && s.current_bitstream_length > 256
        s.current_bitstream_length //= 2
        s.error_correction_enabled = false
end

function encode_ecc(s::ScDoctorState, data)
    if ! s.error_correction_enabled
        return data & 0x0F
    d1 = (data >> 3) & 1
    d2 = (data >> 2) & 1
    d3 = (data >> 1) & 1
    d4 = data & 1
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4
end

function decode_ecc(s::ScDoctorState, encoded)
    if ! s.error_correction_enabled
        return encoded & 0x0F
    p1 = (encoded >> 6) & 1
    p2 = (encoded >> 5) & 1
    d1 = (encoded >> 4) & 1
    p3 = (encoded >> 3) & 1
    d2 = (encoded >> 2) & 1
    d3 = (encoded >> 1) & 1
    d4 = encoded & 1
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1
    corrected = encoded
    bit_positions = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1, 7: 0}
    if syndrome in bit_positions
        corrected ^= (1 << bit_positions[syndrome])
    cd1 = (corrected >> 4) & 1
    cd2 = (corrected >> 2) & 1
    cd3 = (corrected >> 1) & 1
    cd4 = corrected & 1
    return (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4
end

end # module ScDoctorAccel
