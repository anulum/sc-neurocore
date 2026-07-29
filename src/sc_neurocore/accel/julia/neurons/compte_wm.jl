# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compte et al. 2000 midpoint-RK2 Julia lane

"""Source-bounded Compte pyramidal-cell and incoming channel dynamics."""
module CompteWmAccel

export CompteWMNeuronState, get_state, reset!, simulate, simulate_compte_wm!, step!, validate

const GATE_MAX = 1.0e6

"""Complete Compte membrane, channel, refractory, and configuration state."""
mutable struct CompteWMNeuronState
    v::Float64
    s_ampa::Float64
    s_nmda::Float64
    x_nmda::Float64
    s_gaba::Float64
    ref_remaining::Float64
    g_l::Float64
    g_ampa::Float64
    g_nmda::Float64
    g_gaba::Float64
    e_l::Float64
    e_exc::Float64
    e_inh::Float64
    C_m::Float64
    mg::Float64
    tau_ampa::Float64
    tau_nmda::Float64
    tau_x::Float64
    tau_gaba::Float64
    alpha_nmda::Float64
    v_threshold::Float64
    v_reset::Float64
    tau_ref::Float64
    dt::Float64
end

"""Construct the Compte (2000) source control-set pyramidal defaults."""
CompteWMNeuronState() = CompteWMNeuronState(
    -70.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.025, 0.0031, 0.000381, 0.001336, -70.0, 0.0, -70.0, 0.5, 1.0,
    2.0, 100.0, 2.0, 10.0, 0.5, -50.0, -60.0, 2.0, 0.02,
)

"""Return whether every mutable Compte invariant holds."""
function validate(s::CompteWMNeuronState)::Bool
    values = (
        s.v, s.s_ampa, s.s_nmda, s.x_nmda, s.s_gaba, s.ref_remaining,
        s.g_l, s.g_ampa, s.g_nmda, s.g_gaba, s.e_l, s.e_exc, s.e_inh,
        s.C_m, s.mg, s.tau_ampa, s.tau_nmda, s.tau_x, s.tau_gaba,
        s.alpha_nmda, s.v_threshold, s.v_reset, s.tau_ref, s.dt,
    )
    return all(isfinite, values) && -200.0 <= s.v <= 100.0 &&
        -200.0 <= s.v_reset <= 100.0 && 0.0 <= s.s_ampa <= GATE_MAX &&
        0.0 <= s.s_nmda <= 1.0 && 0.0 <= s.x_nmda <= GATE_MAX &&
        0.0 <= s.s_gaba <= GATE_MAX && s.ref_remaining >= 0.0 &&
        all(x -> x >= 0.0, (s.g_l, s.g_ampa, s.g_nmda, s.g_gaba, s.mg, s.alpha_nmda)) &&
        all(x -> x > 0.0, (s.C_m, s.tau_ampa, s.tau_nmda, s.tau_x, s.tau_gaba, s.tau_ref, s.dt))
end

"""Return dynamic state in public trace order."""
get_state(s::CompteWMNeuronState) = (
    s.v, s.s_ampa, s.s_nmda, s.x_nmda, s.s_gaba, s.ref_remaining,
)

function derivatives(s::CompteWMNeuronState, state, current, membrane_active)
    v, s_ampa, s_nmda, x_nmda, s_gaba = state
    d_v = 0.0
    if membrane_active
        block = 1.0 / (1.0 + s.mg / 3.57 * exp(-0.062 * v))
        i_l = s.g_l * (v - s.e_l)
        i_ampa = s.g_ampa * s_ampa * (v - s.e_exc)
        i_nmda = s.g_nmda * block * s_nmda * (v - s.e_exc)
        i_gaba = s.g_gaba * s_gaba * (v - s.e_inh)
        d_v = (-i_l - i_ampa - i_nmda - i_gaba + current) / s.C_m
    end
    result = (
        d_v,
        -s_ampa / s.tau_ampa,
        -s_nmda / s.tau_nmda + s.alpha_nmda * x_nmda * (1.0 - s_nmda),
        -x_nmda / s.tau_x,
        -s_gaba / s.tau_gaba,
    )
    return all(isfinite, result) ? result : nothing
end

"""Advance one atomic midpoint-RK2 step; return -1 without mutation on error."""
function step!(
    s::CompteWMNeuronState,
    current::Float64=0.0;
    recurrent_event::Bool=false,
    external_event::Bool=false,
    inhibitory_event::Bool=false,
)::Int
    if !validate(s) || !isfinite(current)
        return -1
    end
    initial = (
        s.v,
        s.s_ampa + (external_event ? 1.0 : 0.0),
        s.s_nmda,
        s.x_nmda + (recurrent_event ? 1.0 : 0.0),
        s.s_gaba + (inhibitory_event ? 1.0 : 0.0),
    )
    if !all(x -> isfinite(x) && 0.0 <= x <= GATE_MAX, initial[2:5])
        return -1
    end
    active = s.ref_remaining <= 0.0
    k1 = derivatives(s, initial, current, active)
    isnothing(k1) && return -1
    midpoint = ntuple(index -> initial[index] + 0.5 * s.dt * k1[index], 5)
    k2 = derivatives(s, midpoint, current, active)
    isnothing(k2) && return -1
    candidate = ntuple(index -> initial[index] + s.dt * k2[index], 5)
    if !all(isfinite, candidate) || !(-200.0 <= candidate[1] <= 100.0) ||
       !all(x -> 0.0 <= x <= GATE_MAX, candidate[2:5]) || candidate[3] > 1.0
        return -1
    end
    v_next = candidate[1]
    ref_next = max(0.0, s.ref_remaining - s.dt)
    event = 0
    if !active
        v_next = s.v_reset
    elseif v_next >= s.v_threshold
        v_next, ref_next, event = s.v_reset, s.tau_ref, 1
    end
    s.v, s.s_ampa, s.s_nmda, s.x_nmda, s.s_gaba =
        v_next, candidate[2], candidate[3], candidate[4], candidate[5]
    s.ref_remaining = ref_next
    return event
end

"""Reset dynamic state while preserving all configuration."""
function reset!(s::CompteWMNeuronState)
    s.v, s.s_ampa, s.s_nmda, s.x_nmda, s.s_gaba, s.ref_remaining =
        s.e_l, 0.0, 0.0, 0.0, 0.0, 0.0
    return nothing
end

"""Run complete state and event traces over four equal-length input arrays."""
function simulate(state::CompteWMNeuronState, currents, recurrent, external, inhibitory)
    steps = length(currents)
    all(length(values) == steps for values in (recurrent, external, inhibitory)) ||
        throw(ArgumentError("Compte input lengths must match"))
    traces = ntuple(_ -> Vector{Float64}(undef, steps), 6)
    events = Vector{Int64}(undef, steps)
    for index in 1:steps
        event = step!(
            state,
            Float64(currents[index]);
            recurrent_event=recurrent[index] != 0,
            external_event=external[index] != 0,
            inhibitory_event=inhibitory[index] != 0,
        )
        event >= 0 || throw(ArgumentError("invalid Compte batch"))
        dynamic = get_state(state)
        for output in 1:6
            traces[output][index] = dynamic[output]
        end
        events[index] = event
    end
    return traces, events, get_state(state)
end

"""Fill caller-owned complete batch outputs for the Python/Julia boundary."""
function simulate_compte_wm!(
    v, s_ampa, s_nmda, x_nmda, s_gaba, ref_remaining,
    g_l, g_ampa, g_nmda, g_gaba, e_l, e_exc, e_inh, C_m, mg,
    tau_ampa, tau_nmda, tau_x, tau_gaba, alpha_nmda,
    v_threshold, v_reset, tau_ref, dt,
    currents, recurrent, external, inhibitory,
    voltages, s_ampa_out, s_nmda_out, x_nmda_out, s_gaba_out, refractory, events,
)
    state = CompteWMNeuronState(
        Float64(v), Float64(s_ampa), Float64(s_nmda), Float64(x_nmda),
        Float64(s_gaba), Float64(ref_remaining), Float64(g_l), Float64(g_ampa),
        Float64(g_nmda), Float64(g_gaba), Float64(e_l), Float64(e_exc),
        Float64(e_inh), Float64(C_m), Float64(mg), Float64(tau_ampa),
        Float64(tau_nmda), Float64(tau_x), Float64(tau_gaba),
        Float64(alpha_nmda), Float64(v_threshold), Float64(v_reset),
        Float64(tau_ref), Float64(dt),
    )
    validate(state) || throw(ArgumentError("invalid Compte configuration"))
    steps = length(currents)
    all(length(values) == steps for values in (
        recurrent, external, inhibitory, voltages, s_ampa_out, s_nmda_out,
        x_nmda_out, s_gaba_out, refractory, events,
    )) || throw(ArgumentError("Compte input/output lengths must match"))
    outputs = (voltages, s_ampa_out, s_nmda_out, x_nmda_out, s_gaba_out, refractory)
    for index in 1:steps
        event = step!(
            state,
            Float64(currents[index]);
            recurrent_event=recurrent[index] != 0,
            external_event=external[index] != 0,
            inhibitory_event=inhibitory[index] != 0,
        )
        event >= 0 || throw(ArgumentError("invalid Compte batch"))
        dynamic = get_state(state)
        for output in 1:6
            outputs[output][index] = dynamic[output]
        end
        events[index] = event
    end
    return get_state(state)
end

end # module CompteWmAccel
