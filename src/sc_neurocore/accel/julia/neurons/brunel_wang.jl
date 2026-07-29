# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel-Wang 2001 midpoint-RK2 Julia lane

module BrunelWangAccel

export BrunelWangNeuronState, step!, simulate, simulate_brunel_wang!, validate, get_state

"""Complete membrane/configuration state for the Brunel-Wang pyramidal cell."""
mutable struct BrunelWangNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_ref::Float64
    g_ampa_ext::Float64
    g_ampa_rec::Float64
    g_nmda::Float64
    g_gaba::Float64
    v_ampa::Float64
    v_nmda::Float64
    v_gaba::Float64
    C_m::Float64
    mg_conc::Float64
    dt::Float64
    ref_remaining::Float64
end

"""Construct Brunel and Wang's excitatory pyramidal-cell defaults."""
BrunelWangNeuronState() = BrunelWangNeuronState(
    -70.0, -70.0, -55.0, -50.0, 20.0, 2.0, 2.08, 0.104, 0.327, 1.25,
    0.0, 0.0, -70.0, 0.5, 1.0, 0.1, 0.0,
)

nonnegative(x::Float64)::Bool = isfinite(x) && x >= 0.0

"""Return whether every runtime/configuration invariant holds."""
function validate(s::BrunelWangNeuronState)::Bool
    values = (
        s.v, s.v_rest, s.v_reset, s.v_threshold, s.tau_m, s.tau_ref,
        s.g_ampa_ext, s.g_ampa_rec, s.g_nmda, s.g_gaba, s.v_ampa,
        s.v_nmda, s.v_gaba, s.C_m, s.mg_conc, s.dt, s.ref_remaining,
    )
    return all(isfinite, values) && s.tau_m > 0.0 && s.tau_ref > 0.0 &&
        s.C_m > 0.0 && s.dt > 0.0 && s.g_ampa_ext >= 0.0 &&
        s.g_ampa_rec >= 0.0 && s.g_nmda >= 0.0 && s.g_gaba >= 0.0 &&
        s.mg_conc >= 0.0 && s.ref_remaining >= 0.0
end

"""Return the complete membrane/refractory state."""
get_state(s::BrunelWangNeuronState) = (v=s.v, ref_remaining=s.ref_remaining)

function derivative(s::BrunelWangNeuronState, v, ext, ampa, nmda, gaba)
    block = 1.0 / (1.0 + s.mg_conc / 3.57 * exp(-0.062 * v))
    i_ampa = -s.g_ampa_ext * (v - s.v_ampa) * ext - s.g_ampa_rec * (v - s.v_ampa) * ampa
    i_nmda = -s.g_nmda * block * (v - s.v_nmda) * nmda
    i_gaba = -s.g_gaba * (v - s.v_gaba) * gaba
    return -(v - s.v_rest) / s.tau_m + (i_ampa + i_nmda + i_gaba) / s.C_m
end

"""Advance one atomic midpoint-RK2 step; return -1 without mutation on failure."""
function step!(s::BrunelWangNeuronState, ext::Float64, ampa::Float64, nmda::Float64, gaba::Float64)::Int
    if !validate(s) || !all(nonnegative, (ext, ampa, nmda, gaba))
        return -1
    end
    if s.ref_remaining > 0.0
        s.v = s.v_reset
        s.ref_remaining = max(0.0, s.ref_remaining - s.dt)
        return 0
    end
    v = s.v
    k1 = derivative(s, v, ext, ampa, nmda, gaba)
    midpoint = v + 0.5 * s.dt * k1
    k2 = derivative(s, midpoint, ext, ampa, nmda, gaba)
    candidate = v + s.dt * k2
    if !all(isfinite, (k1, midpoint, k2, candidate))
        return -1
    end
    s.v = candidate
    if candidate >= s.v_threshold
        s.v = s.v_reset
        s.ref_remaining = s.tau_ref
        return 1
    end
    return 0
end

"""Run complete four-gate traces and final state without surrogate fallback."""
function simulate(s::BrunelWangNeuronState, ext, ampa, nmda, gaba)
    n = length(ext)
    length(ampa) == n && length(nmda) == n && length(gaba) == n ||
        throw(ArgumentError("Brunel-Wang input lengths must match"))
    voltages = Vector{Float64}(undef, n)
    refractory = Vector{Float64}(undef, n)
    events = Vector{Int64}(undef, n)
    for index in 1:n
        event = step!(s, Float64(ext[index]), Float64(ampa[index]), Float64(nmda[index]), Float64(gaba[index]))
        event >= 0 || throw(ArgumentError("invalid Brunel-Wang batch"))
        voltages[index] = s.v
        refractory[index] = s.ref_remaining
        events[index] = event
    end
    return Dict(
        "voltages" => voltages,
        "refractory" => refractory,
        "events" => events,
        "v_final" => s.v,
        "ref_final" => s.ref_remaining,
    )
end

"""Batch facade with caller-owned outputs for the Python/Julia boundary."""
function simulate_brunel_wang!(
    v, ref_remaining, v_rest, v_reset, v_threshold, tau_m, tau_ref,
    g_ampa_ext, g_ampa_rec, g_nmda, g_gaba, v_ampa, v_nmda, v_gaba,
    C_m, mg_conc, dt, ext, ampa, nmda, gaba, voltages, refractory, events,
)
    state = BrunelWangNeuronState(
        Float64(v), Float64(v_rest), Float64(v_reset), Float64(v_threshold),
        Float64(tau_m), Float64(tau_ref), Float64(g_ampa_ext), Float64(g_ampa_rec),
        Float64(g_nmda), Float64(g_gaba), Float64(v_ampa), Float64(v_nmda),
        Float64(v_gaba), Float64(C_m), Float64(mg_conc), Float64(dt),
        Float64(ref_remaining),
    )
    n = length(ext)
    length(ampa) == n && length(nmda) == n && length(gaba) == n ||
        throw(ArgumentError("Brunel-Wang input lengths must match"))
    length(voltages) == n && length(refractory) == n && length(events) == n ||
        throw(ArgumentError("Brunel-Wang output lengths must match"))
    @inbounds for index in 1:n
        event = step!(state, Float64(ext[index]), Float64(ampa[index]), Float64(nmda[index]), Float64(gaba[index]))
        event >= 0 || throw(ArgumentError("invalid Brunel-Wang batch"))
        voltages[index] = state.v
        refractory[index] = state.ref_remaining
        events[index] = event
    end
    return (state.v, state.ref_remaining)
end

end # module BrunelWangAccel
