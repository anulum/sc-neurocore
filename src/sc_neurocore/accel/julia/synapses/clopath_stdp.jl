# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/clopath_stdp

module ClopathStdpAccel

using Statistics, LinearAlgebra

mutable struct ClopathSTDPState
    a_ltd::Float64
    a_ltp::Float64
    tau_x::Float64
    tau_minus::Float64
    tau_plus::Float64
    theta_minus::Float64
    theta_plus::Float64
    w_min::Float64
    w_max::Float64
    weight::Float64
end

function ClopathSTDPState()
    ClopathSTDPState(0.00014, 8e-05, 15.0, 10.0, 7.0, -70.6, -45.3, 0.0, 1.0, 0.5)
end

function step(s::ClopathSTDPState, pre_spike, u_post, dt)
    decay_x = math.exp(-dt / s.tau_x)
    decay_minus = math.exp(-dt / s.tau_minus)
    decay_plus = math.exp(-dt / s.tau_plus)
    # LTD: pre-synaptic spike × post depolarization (Clopath 2010, Eq. 2)
    if pre_spike
        ltd = s.a_ltd * s.x_bar * max(0.0, s.u_bar_minus - s.theta_minus)
        s.weight -= ltd
    # LTP: evaluated every timestep, pre contribution via x_bar trace (Clopath 2010, Eq. 3)
    ltp_post = max(0.0, u_post - s.theta_plus)
    ltp_pre = max(0.0, s.u_bar_plus - s.theta_minus)
    if ltp_post > 0 && ltp_pre > 0
        s.weight += s.a_ltp * s.x_bar * ltp_post * ltp_pre
    s.weight = max(s.w_min, min(s.w_max, s.weight))
    # Update traces: exact exponential filter (no double-decay)
    s.x_bar *= decay_x
    if pre_spike
        s.x_bar += 1.0
    s.u_bar_minus = decay_minus * s.u_bar_minus + (1 - decay_minus) * u_post
    s.u_bar_plus = decay_plus * s.u_bar_plus + (1 - decay_plus) * u_post
    return s.weight
end

function reset(s::ClopathSTDPState)
    s.x_bar = 0.0
    s.u_bar_minus = 0.0
    s.u_bar_plus = 0.0
end

end # module ClopathStdpAccel
