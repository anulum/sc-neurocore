# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for compte_wm

module CompteWmAccel

export step!, simulate, CompteWMNeuronState, validate

const V_MIN = -200.0
const V_MAX = 100.0
const GATE_MAX = 1.0e6
const GABA_TAU = 5.0

mutable struct CompteWMNeuronState
    v::Float64
    s_ampa::Float64
    s_nmda::Float64
    x_nmda::Float64
    s_gaba::Float64
    g_l::Float64
    g_ampa::Float64
    g_nmda::Float64
    g_gaba::Float64
    e_l::Float64
    e_exc::Float64
    e_inh::Float64
    c_m::Float64
    mg::Float64
    tau_ampa::Float64
    tau_nmda::Float64
    tau_x::Float64
    alpha_nmda::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function CompteWMNeuronState()
    CompteWMNeuronState(-70.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.005, 0.165, 0.013, -70.0, 0.0, -70.0, 0.5, 1.0, 2.0, 100.0, 2.0, 0.5, -50.0, -55.0, 0.1)
end

_finite(value::Float64) = isfinite(value)
_gate(value::Float64) = isfinite(value) && 0.0 <= value <= GATE_MAX

function _decay(dt::Float64, tau::Float64)
    ratio = -dt / tau
    decay = ratio < -700.0 ? 0.0 : exp(ratio)
    if !_finite(decay) || decay < 0.0 || decay >= 1.0
        error("decay must be in [0, 1)")
    end
    return decay
end

function validate(s::CompteWMNeuronState)
    if !_finite(s.v) || !(V_MIN <= s.v <= V_MAX)
        return false
    end
    if !_gate(s.s_ampa) || !_gate(s.s_nmda) || !_gate(s.x_nmda) || !_gate(s.s_gaba) || s.s_nmda > 1.0
        return false
    end
    if any(x -> !_finite(x) || x < 0.0, (s.g_l, s.g_ampa, s.g_nmda, s.g_gaba, s.mg, s.alpha_nmda))
        return false
    end
    if any(x -> !_finite(x) || x <= 0.0, (s.c_m, s.tau_ampa, s.tau_nmda, s.tau_x, s.dt))
        return false
    end
    if any(x -> !_finite(x), (s.e_l, s.e_exc, s.e_inh, s.v_threshold, s.v_reset))
        return false
    end
    return V_MIN <= s.v_reset <= V_MAX
end

function _mg_block(s::CompteWMNeuronState, v::Float64)
    exponent = -0.062 * v
    exp_value = exponent < -700.0 ? 0.0 : exp(min(exponent, 700.0))
    denominator = 1.0 + s.mg / 3.57 * exp_value
    if !_finite(denominator) || denominator <= 0.0
        error("Mg block denominator invalid")
    end
    block = 1.0 / denominator
    if block < 0.0 || block > 1.0
        error("Mg block outside [0, 1]")
    end
    return block
end

function step!(s::CompteWMNeuronState, I_ext::Float64=0.0; spike_in::Bool=false, dt::Float64=s.dt)
    if !_finite(I_ext) || !validate(s)
        return -1
    end
    decay_ampa = _decay(s.dt, s.tau_ampa)
    decay_x = _decay(s.dt, s.tau_x)
    decay_gaba = _decay(s.dt, GABA_TAU)
    spike_increment = spike_in ? 1.0 : 0.0
    s_ampa_pre = s.s_ampa + spike_increment
    x_nmda_pre = s.x_nmda + spike_increment
    if s_ampa_pre > GATE_MAX || x_nmda_pre > GATE_MAX
        return -1
    end
    s_ampa_candidate = s_ampa_pre * decay_ampa
    s_nmda_candidate = s.s_nmda + (-s.s_nmda / s.tau_nmda + s.alpha_nmda * x_nmda_pre * (1.0 - s.s_nmda)) * s.dt
    x_nmda_candidate = x_nmda_pre * decay_x
    s_gaba_candidate = s.s_gaba * decay_gaba
    if any(x -> !_finite(x) || x < 0.0 || x > GATE_MAX, (s_ampa_candidate, s_nmda_candidate, x_nmda_candidate, s_gaba_candidate)) || s_nmda_candidate > 1.0
        return -1
    end
    b = _mg_block(s, s.v)
    i_l = s.g_l * (s.v - s.e_l)
    i_ampa = s.g_ampa * s_ampa_candidate * (s.v - s.e_exc)
    i_nmda = s.g_nmda * b * s_nmda_candidate * (s.v - s.e_exc)
    i_gaba = s.g_gaba * s_gaba_candidate * (s.v - s.e_inh)
    dv = (-i_l - i_ampa - i_nmda - i_gaba + I_ext) / s.c_m * s.dt
    v_candidate = s.v + dv
    if any(x -> !_finite(x), (i_l, i_ampa, i_nmda, i_gaba, dv, v_candidate)) || !(V_MIN <= v_candidate <= V_MAX)
        return -1
    end
    if v_candidate >= s.v_threshold
        gaba_after_spike = s_gaba_candidate + 1.0
        if gaba_after_spike > GATE_MAX
            return -1
        end
        s.v = s.v_reset
        s.s_ampa = s_ampa_candidate
        s.s_nmda = s_nmda_candidate
        s.x_nmda = x_nmda_candidate
        s.s_gaba = gaba_after_spike
        return 1
    end
    s.v = v_candidate
    s.s_ampa = s_ampa_candidate
    s.s_nmda = s_nmda_candidate
    s.x_nmda = x_nmda_candidate
    s.s_gaba = s_gaba_candidate
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = CompteWMNeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = result < 0 ? NaN : s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module CompteWmAccel
