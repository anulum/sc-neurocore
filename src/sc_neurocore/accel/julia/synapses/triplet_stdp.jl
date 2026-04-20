# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/triplet_stdp

module TripletStdpAccel

using Statistics, LinearAlgebra

mutable struct TripletSTDPState
    tau_plus::Float64
    tau_minus::Float64
    tau_x::Float64
    tau_y::Float64
    a2_plus::Float64
    a3_plus::Float64
    a2_minus::Float64
    a3_minus::Float64
    w_min::Float64
    w_max::Float64
    weight::Float64
end

function TripletSTDPState()
    TripletSTDPState(16.8, 33.7, 101.0, 125.0, 7.5e-10, 0.0093, 0.007, 0.00023, 0.0, 1.0, 0.5)
end

function step(s::TripletSTDPState, pre_spike, post_spike, dt)
    import math
    # Decay traces
    s.r1 *= math.exp(-dt / s.tau_plus)
    s.r2 *= math.exp(-dt / s.tau_x)
    s.o1 *= math.exp(-dt / s.tau_minus)
    s.o2 *= math.exp(-dt / s.tau_y)
    # Weight updates on spikes
    if post_spike
        # LTP: pair + triplet pre-post-post
        s.weight += s.r1 * (s.a2_plus + s.a3_plus * s.o2)
    if pre_spike
        # LTD: pair + triplet pre-pre-post
        s.weight -= s.o1 * (s.a2_minus + s.a3_minus * s.r2)
    # Clamp
    s.weight = max(s.w_min, min(s.w_max, s.weight))
    # Update traces after weight change (order matters — Pfister 2006 Eq. 3-4)
    if pre_spike
        s.r1 += 1.0
        s.r2 += 1.0
    if post_spike
        s.o1 += 1.0
        s.o2 += 1.0
    return s.weight
end

function reset(s::TripletSTDPState)
    s.r1 = s.r2 = s.o1 = s.o2 = 0.0
end

end # module TripletStdpAccel
