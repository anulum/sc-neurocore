# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for astrocyte

module AstrocyteAccel

export step!, simulate, AstrocyteModelState

mutable struct AstrocyteModelState
    ca::Float64
    h::Float64
    ip3::Float64
    v_er::Float64
    k_er::Float64
    v_serca::Float64
    d1::Float64
    d2::Float64
    d3::Float64
    d5::Float64
    a2::Float64
    c0::Float64
    c1::Float64
    leak::Float64
    ip3_prod::Float64
    ip3_decay::Float64
    dt::Float64
end

function AstrocyteModelState()
    AstrocyteModelState(0.05, 0.8, 0.5, 0.9, 0.15, 0.4, 0.13, 1.049, 0.9434, 0.08234, 0.2, 2.0, 0.185, 0.01, 0.0, 0.14, 0.01)
end

function valid_runtime(s::AstrocyteModelState)
    positive = (s.v_er, s.k_er, s.v_serca, s.d1, s.d2, s.d3, s.d5, s.a2, s.c0, s.c1, s.dt)
    non_negative = (s.leak, s.ip3_prod, s.ip3_decay)
    all(x -> isfinite(x) && x > 0.0, positive) &&
        all(x -> isfinite(x) && x >= 0.0, non_negative) &&
        isfinite(s.ca) && 0.0 <= s.ca <= s.c0 &&
        isfinite(s.h) && 0.0 <= s.h <= 1.0 &&
        isfinite(s.ip3) && s.ip3 >= 0.0
end

"""
    step!(s, I_ext=0.0)

Advance one forward-Euler step of the Li-Rinzel recurrence with `I_ext`, the input IP3
production (uM/s), and return the new cytosolic calcium. The recurrence is the Python
reference's, term for term. A negative or non-finite input, an invalid parameter or state, or
a calcium candidate that is not finite or exceeds `c0` returns `nothing` and leaves the state
unchanged.
"""
function step!(s::AstrocyteModelState, I_ext::Float64=0.0)
    (isfinite(I_ext) && I_ext >= 0.0) || return nothing
    valid_runtime(s) || return nothing
    m_inf = s.ip3 / (s.ip3 + s.d1)
    n_inf = s.ca / (s.ca + s.d5)
    ca_er = (s.c0 - s.ca) / s.c1
    j_channel = s.v_er * (m_inf * n_inf * s.h) ^ 3 * (ca_er - s.ca)
    j_serca = s.v_serca * s.ca ^ 2 / (s.ca ^ 2 + s.k_er ^ 2)
    j_leak = s.leak * (ca_er - s.ca)
    dca = j_channel - j_serca + j_leak
    q2 = s.d2 * (s.ip3 + s.d1) / (s.ip3 + s.d3)
    h_inf = q2 / (q2 + s.ca)
    tau_h = 1.0 / (s.a2 * (q2 + s.ca))
    dh = (h_inf - s.h) / max(tau_h, 1e-06)
    dip3 = I_ext + s.ip3_prod - s.ip3_decay * s.ip3
    ca_next = s.ca + dca * s.dt
    (isfinite(ca_next) && ca_next <= s.c0) || return nothing
    s.ca = max(0.0, ca_next)
    s.h = clamp(s.h + dh * s.dt, 0.0, 1.0)
    s.ip3 = max(0.0, s.ip3 + dip3 * s.dt)
    return s.ca
end

"""
    simulate(n_steps=1000; I_ext=10.0)

Run `n_steps` steps of a default cell at a constant input and return the calcium trace and
the number of refused steps.
"""
function simulate(n_steps::Int=1000; I_ext::Float64=10.0)
    s = AstrocyteModelState()
    trace = zeros(n_steps)
    refused = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        result === nothing && (refused += 1)
        trace[t] = s.ca
    end
    return trace, refused
end

end # module AstrocyteAccel
