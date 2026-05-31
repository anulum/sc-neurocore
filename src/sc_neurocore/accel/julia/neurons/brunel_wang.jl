# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for brunel_wang

module BrunelWangAccel

export step!, simulate, validate, get_state, BrunelWangNeuronState

mutable struct BrunelWangNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_ref::Float64
    tau_ampa::Float64
    tau_nmda_rise::Float64
    tau_nmda_decay::Float64
    tau_gaba::Float64
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

function BrunelWangNeuronState()
    BrunelWangNeuronState(-70.0, -70.0, -55.0, -50.0, 20.0, 2.0, 2.0, 2.0, 100.0, 5.0, 2.1, 0.05, 0.165, 1.3, 0.0, 0.0, -70.0, 0.5, 1.0, 0.1, 0.0)
end

positive(x::Float64)::Bool = isfinite(x) && x > 0.0
nonnegative(x::Float64)::Bool = isfinite(x) && x >= 0.0
gate(x::Float64)::Bool = isfinite(x) && 0.0 <= x <= 1.0

function validate(s::BrunelWangNeuronState)::Bool
    return isfinite(s.v) &&
        isfinite(s.v_rest) &&
        isfinite(s.v_reset) &&
        isfinite(s.v_threshold) &&
        positive(s.tau_m) &&
        positive(s.tau_ref) &&
        positive(s.tau_ampa) &&
        positive(s.tau_nmda_rise) &&
        positive(s.tau_nmda_decay) &&
        positive(s.tau_gaba) &&
        nonnegative(s.g_ampa_ext) &&
        nonnegative(s.g_ampa_rec) &&
        nonnegative(s.g_nmda) &&
        nonnegative(s.g_gaba) &&
        isfinite(s.v_ampa) &&
        isfinite(s.v_nmda) &&
        isfinite(s.v_gaba) &&
        positive(s.C_m) &&
        nonnegative(s.mg_conc) &&
        positive(s.dt) &&
        nonnegative(s.ref_remaining)
end

function _nmda_voltage_dep(s::BrunelWangNeuronState, v)
    isfinite(v) || throw(ArgumentError("voltage must be finite"))
    exponent = -0.062 * v
    if exponent > 700.0
        return 0.0
    end
    factor = 1.0 / (1.0 + s.mg_conc / 3.57 * exp(exponent))
    isfinite(factor) && 0.0 <= factor <= 1.0 ||
        throw(ArgumentError("invalid Brunel-Wang NMDA voltage factor"))
    return factor
end

function get_state(s::BrunelWangNeuronState)
    return (v=s.v, ref_remaining=s.ref_remaining)
end

function step!(
    s::BrunelWangNeuronState,
    i_ampa_ext::Float64=0.0,
    s_ampa_rec::Float64=0.0,
    s_nmda_rec::Float64=0.0,
    s_gaba::Float64=0.0;
    dt::Float64=0.1,
)
    if !validate(s) || !nonnegative(i_ampa_ext) || !gate(s_ampa_rec) || !gate(s_nmda_rec) || !gate(s_gaba)
        return -1
    end
    if s.ref_remaining > 0.0
        s.ref_remaining = max(0.0, s.ref_remaining - s.dt)
        return 0
    end

    i_ampa = -s.g_ampa_ext * (s.v - s.v_ampa) * i_ampa_ext
    i_ampa += -s.g_ampa_rec * (s.v - s.v_ampa) * s_ampa_rec
    i_nmda = -s.g_nmda * _nmda_voltage_dep(s, s.v) * (s.v - s.v_nmda) * s_nmda_rec
    i_gaba = -s.g_gaba * (s.v - s.v_gaba) * s_gaba
    i_leak = -(s.v - s.v_rest) / s.tau_m
    dv = (i_leak + (i_ampa + i_nmda + i_gaba) / s.C_m) * s.dt
    next_v = s.v + dv
    if !all(isfinite, (i_ampa, i_nmda, i_gaba, i_leak, dv, next_v))
        return -1
    end
    s.v = next_v
    if s.v >= s.v_threshold
        s.v = s.v_reset
        s.ref_remaining = s.tau_ref
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BrunelWangNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module BrunelWangAccel
