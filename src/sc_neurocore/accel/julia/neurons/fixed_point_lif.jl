# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fixed_point_lif

module FixedPointLifAccel

export step!, simulate, FixedPointLIFState

mutable struct FixedPointLIFState
    data_width::Int
    fraction::Int
    v_rest::Int
    v_reset::Int
    v_threshold::Int
    refractory_period::Int
    width::Int
    seed::UInt32
    reg::UInt32
    v::Int
    ref_count::Int
end

function FixedPointLIFState()
    w = 8; frac = 4
    scale = 1 << frac
    FixedPointLIFState(w, frac, -65*scale, -65*scale, -50*scale, 3, w, UInt32(0xACE1), UInt32(0xACE1), -65*scale, 0)
end

function lfsr_step!(s::FixedPointLIFState)
    bit = (s.reg >> (s.width - 1)) ⊻ (s.reg >> (s.width - 2)) ⊻ (s.reg >> (s.width - 3)) ⊻ (s.reg >> (s.width - 5))
    s.reg = ((s.reg << 1) | (bit & UInt32(1))) & ((UInt32(1) << s.width) - UInt32(1))
    return s.reg
end

function step!(s::FixedPointLIFState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if s.ref_count > 0
            s.ref_count -= 1
            return 0
        end
        scale = 1 << s.fraction
        i_fixed = round(Int, I_ext * scale)
        decay = s.v >> 2
        s.v = s.v - decay + i_fixed
        noise = Int(lfsr_step!(s) & UInt32(0x0F)) - 8
        s.v += noise
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.ref_count = s.refractory_period
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FixedPointLIFState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = Float64(s.v) / (1 << s.fraction)
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module FixedPointLifAccel
