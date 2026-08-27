# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for TwoCompartmentLIFNeuron

module TcLifAccel

export step!, simulate, reset!, valid, TwoCompartmentLIFNeuronState

"""TC-LIF map state (Zhang et al. 2024, Eqs. 10-12) mirroring the Python reference."""
mutable struct TwoCompartmentLIFNeuronState
    u_d::Float64
    u_s::Float64
    s_prev::Float64
    beta1::Float64
    beta2::Float64
    gamma::Float64
    v_th::Float64
end

function TwoCompartmentLIFNeuronState()
    TwoCompartmentLIFNeuronState(0.0, 0.0, 0.0, -0.5, 0.5, 0.5, 1.0)
end

"""Return whether every state and configuration field is finite and inside the public bounds."""
function valid(s::TwoCompartmentLIFNeuronState)
    values = (s.u_d, s.u_s, s.s_prev, s.beta1, s.beta2, s.gamma, s.v_th)
    all(isfinite, values) &&
        -1e6 <= s.u_d <= 1e6 && -1e6 <= s.u_s <= 1e6 &&
        (s.s_prev == 0.0 || s.s_prev == 1.0) &&
        -1.0 < s.beta1 < 0.0 && 0.0 < s.beta2 < 1.0 &&
        0.0 <= s.gamma <= 10.0 && 0.0 < s.v_th <= 100.0
end

"""
    step!(state, i_ext) -> Int

Advance the TC-LIF map one step (U_D -> U_S -> S ordering, delayed
soft reset through S[t-1]) and return the spike indicator. Throws
`ArgumentError` — with the pre-step state preserved exactly — for a
non-finite input, an out-of-bounds configuration, or a non-finite
candidate.
"""
function step!(s::TwoCompartmentLIFNeuronState, i_ext::Float64)
    isfinite(i_ext) || throw(ArgumentError("i_ext must be finite"))
    valid(s) || throw(
        ArgumentError("TC-LIF state and parameters must satisfy the public bounds")
    )

    u_d = s.u_d + s.beta1 * s.u_s + i_ext - s.gamma * s.s_prev
    u_s = s.u_s + s.beta2 * u_d - s.v_th * s.s_prev
    (isfinite(u_d) && isfinite(u_s)) ||
        throw(ArgumentError("TC-LIF candidate state became non-finite"))
    spike = u_s >= s.v_th ? 1 : 0

    s.u_d = u_d
    s.u_s = u_s
    s.s_prev = Float64(spike)
    spike
end

"""Restore the dynamic state to zero, preserving configuration."""
function reset!(s::TwoCompartmentLIFNeuronState)
    s.u_d, s.u_s, s.s_prev = 0.0, 0.0, 0.0
    nothing
end

"""Run a fresh default-profile state for `n_steps` and return `(trace, spikes)`."""
function simulate(n_steps::Int=1000; I_ext::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = TwoCompartmentLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        spikes += step!(s, I_ext)
        trace[t] = s.u_s
    end
    trace, spikes
end

end # module TcLifAccel
