# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for alpha

module AlphaAccel

export step!, simulate, AlphaNeuronState

mutable struct AlphaNeuronState
    v::Float64
    a_exc::Float64
    i_exc::Float64
    a_inh::Float64
    i_inh::Float64
    v_rest::Float64
    v_threshold::Float64
    tau_v::Float64
    tau_exc::Float64
    tau_inh::Float64
    dt::Float64
end

function AlphaNeuronState()
    AlphaNeuronState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)
end

function alpha_finite(value::Float64)::Bool
    isfinite(value)
end

function validate_alpha(s::AlphaNeuronState, step_dt::Float64)::Nothing
    if !(alpha_finite(s.v) && alpha_finite(s.a_exc) && alpha_finite(s.i_exc) &&
         alpha_finite(s.a_inh) && alpha_finite(s.i_inh) &&
         alpha_finite(s.v_rest) && alpha_finite(s.v_threshold))
        error("alpha state variables must be finite")
    end
    if !(alpha_finite(s.tau_v) && s.tau_v > 0.0 &&
         alpha_finite(s.tau_exc) && s.tau_exc > 0.0 &&
         alpha_finite(s.tau_inh) && s.tau_inh > 0.0 &&
         alpha_finite(step_dt) && step_dt > 0.0)
        error("alpha time constants and timestep must be finite and positive")
    end
    return nothing
end

function alpha_filter_candidates(rise_state::Float64, current_state::Float64,
                                 drive::Float64, tau::Float64,
                                 step_dt::Float64)::Tuple{Float64, Float64}
    steady_state = tau * drive
    rise_delta = rise_state - steady_state
    current_delta = current_state - steady_state
    decay = exp(-step_dt / tau)
    rise_next = steady_state + rise_delta * decay
    current_next = steady_state + decay * (current_delta + rise_delta * step_dt / tau)
    return rise_next, current_next
end

function alpha_membrane_drive_contribution(current_delta::Float64, rise_delta::Float64,
                                           tau_drive::Float64, tau_v::Float64,
                                           step_dt::Float64)::Float64
    rate_v = 1.0 / tau_v
    rate_drive = 1.0 / tau_drive
    decay_v = exp(-step_dt / tau_v)
    decay_drive = exp(-step_dt / tau_drive)
    if abs(rate_v - rate_drive) <= 1.0e-14
        return rate_v * decay_v * (
            current_delta * step_dt + rise_delta * step_dt * step_dt / (2.0 * tau_drive)
        )
    end
    rate_delta = rate_v - rate_drive
    first_order = current_delta * (decay_drive - decay_v) / rate_delta
    second_order = (
        rise_delta / tau_drive *
        (decay_drive * (rate_delta * step_dt - 1.0) + decay_v) /
        (rate_delta * rate_delta)
    )
    return rate_v * (first_order + second_order)
end

function step!(s::AlphaNeuronState, exc_current::Float64=0.0,
               inh_current::Float64=0.0; dt::Float64=s.dt)::Int64
    if !(alpha_finite(exc_current) && alpha_finite(inh_current))
        error("alpha currents must be finite")
    end
    validate_alpha(s, dt)

    exc_steady = s.tau_exc * exc_current
    inh_steady = s.tau_inh * inh_current
    exc_rise_delta = s.a_exc - exc_steady
    inh_rise_delta = s.a_inh - inh_steady
    exc_current_delta = s.i_exc - exc_steady
    inh_current_delta = s.i_inh - inh_steady
    a_exc_next, i_exc_next = alpha_filter_candidates(
        s.a_exc, s.i_exc, exc_current, s.tau_exc, dt
    )
    a_inh_next, i_inh_next = alpha_filter_candidates(
        s.a_inh, s.i_inh, inh_current, s.tau_inh, dt
    )
    v_steady = s.v_rest + exc_steady - inh_steady
    v_next = (
        v_steady +
        (s.v - v_steady) * exp(-dt / s.tau_v) +
        alpha_membrane_drive_contribution(
            exc_current_delta, exc_rise_delta, s.tau_exc, s.tau_v, dt
        ) -
        alpha_membrane_drive_contribution(
            inh_current_delta, inh_rise_delta, s.tau_inh, s.tau_v, dt
        )
    )
    if !(alpha_finite(a_exc_next) && alpha_finite(i_exc_next) &&
         alpha_finite(a_inh_next) && alpha_finite(i_inh_next) && alpha_finite(v_next))
        error("alpha exact-flow update became non-finite")
    end

    s.a_exc = a_exc_next
    s.i_exc = i_exc_next
    s.a_inh = a_inh_next
    s.i_inh = i_inh_next
    if s.v_threshold <= v_next
        s.v = s.v_rest
        return 1
    end
    s.v = v_next
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AlphaNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext, 0.0; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AlphaAccel
