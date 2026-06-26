# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 for neurogrid

module NeurogridAccel

export step!, simulate, NeuroGridNeuronState

mutable struct NeuroGridNeuronState
    v_s::Float64
    v_d::Float64
    tau_s::Float64
    tau_d::Float64
    g_c::Float64
    delta_t::Float64
    v_rest::Float64
    v_threshold::Float64
    v_peak::Float64
    v_reset::Float64
    dt::Float64
end

function NeuroGridNeuronState()
    NeuroGridNeuronState(-65.0, -65.0, 20.0, 50.0, 0.5, 2.0, -65.0, -50.0, 20.0, -65.0, 0.1)
end

@inline function _finite(values...)
    return all(isfinite, values)
end

@inline function _valid(s::NeuroGridNeuronState)
    return _finite(s.v_s, s.v_d, s.tau_s, s.tau_d, s.g_c, s.delta_t, s.v_rest, s.v_threshold, s.v_peak, s.v_reset, s.dt) &&
        s.tau_s > 0.0 && s.tau_d > 0.0 && s.delta_t > 0.0 && s.dt > 0.0 && s.g_c >= 0.0
end

@inline function _derivatives(s::NeuroGridNeuronState, v_s::Float64, v_d::Float64, current::Float64)
    v_s_eff = min(v_s, s.v_peak)
    dv_d = (-(v_d - s.v_rest) + current - s.g_c * (v_d - v_s_eff)) / s.tau_d
    exp_arg = min((v_s_eff - s.v_threshold) / s.delta_t, 20.0)
    exp_term = s.delta_t * exp(exp_arg)
    dv_s = (-(v_s_eff - s.v_rest) + exp_term + s.g_c * (v_d - v_s_eff)) / s.tau_s
    return dv_s, dv_d
end

@inline function _rk4_substep(s::NeuroGridNeuronState, v_s::Float64, v_d::Float64, current::Float64)
    dt = s.dt
    k1s, k1d = _derivatives(s, v_s, v_d, current)
    k2s, k2d = _derivatives(s, v_s + 0.5 * dt * k1s, v_d + 0.5 * dt * k1d, current)
    k3s, k3d = _derivatives(s, v_s + 0.5 * dt * k2s, v_d + 0.5 * dt * k2d, current)
    k4s, k4d = _derivatives(s, v_s + dt * k3s, v_d + dt * k3d, current)
    return (
        v_s + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
        v_d + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
    )
end

function step!(s::NeuroGridNeuronState, I_ext::Float64=0.0)
    if !_finite(I_ext) || !_valid(s)
        return 0
    end
    next_v_s, next_v_d = _rk4_substep(s, s.v_s, s.v_d, I_ext)
    if !_finite(next_v_s, next_v_d)
        return 0
    end
    s.v_d = next_v_d
    if next_v_s >= s.v_peak
        s.v_s = s.v_reset
        return 1
    end
    s.v_s = next_v_s
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=100.0)
    s = NeuroGridNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module NeurogridAccel
