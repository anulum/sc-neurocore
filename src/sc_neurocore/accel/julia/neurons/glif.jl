# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Teeter 2018 GLIF5 source model

# Five-state exact-flow specialization of Teeter et al. (2018), Eqs. 1–8.
# The pre-step ASCs are held during the membrane and voltage-threshold flow,
# matching the official AllenSDK exact-dynamics update order.

module GLIF5Accel

export simulate_trace

decay(rate, dt) = exp(-rate * dt)

function exponential_convolution(decay_rate, forcing_rate, dt)
    difference = decay_rate - forcing_rate
    scale = max(1.0, abs(decay_rate), abs(forcing_rate))
    abs(difference) <= 1.0e-12 * scale && return dt * exp(-decay_rate * dt)
    return (exp(-forcing_rate * dt) - exp(-decay_rate * dt)) / difference
end

"""Run a failure-atomic constant-current GLIF5 batch."""
function simulate_trace(
    v0::Float64,
    theta_spike0::Float64,
    i_asc1_0::Float64,
    i_asc2_0::Float64,
    theta_voltage0::Float64,
    refractory_remaining0::Float64,
    e_l::Float64,
    capacitance::Float64,
    resistance::Float64,
    theta_inf::Float64,
    b_spike::Float64,
    b_voltage::Float64,
    a_voltage::Float64,
    k_asc1::Float64,
    k_asc2::Float64,
    f_v::Float64,
    delta_v::Float64,
    delta_theta_spike::Float64,
    f_asc1::Float64,
    f_asc2::Float64,
    delta_i_asc1::Float64,
    delta_i_asc2::Float64,
    refractory_period::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    values = (
        v0, theta_spike0, i_asc1_0, i_asc2_0, theta_voltage0,
        refractory_remaining0, e_l, capacitance, resistance, theta_inf,
        b_spike, b_voltage, a_voltage, k_asc1, k_asc2, f_v, delta_v,
        delta_theta_spike, f_asc1, f_asc2, delta_i_asc1, delta_i_asc2,
        refractory_period, dt, current,
    )
    all(isfinite, values) || throw(ArgumentError("state, parameters and current must be finite"))
    all(>(0.0), (capacitance, resistance, b_spike, b_voltage, k_asc1, k_asc2, dt)) ||
        throw(ArgumentError("time constants, resistance, capacitance and dt must be positive"))
    refractory_remaining0 >= 0.0 || throw(ArgumentError("refractory state must be non-negative"))
    refractory_period >= 0.0 || throw(ArgumentError("refractory period must be non-negative"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))

    trace = Vector{Float64}(undef, n_steps)
    v = v0
    theta_spike = theta_spike0
    i_asc1 = i_asc1_0
    i_asc2 = i_asc2_0
    theta_voltage = theta_voltage0
    refractory_remaining = refractory_remaining0
    events = 0
    membrane_rate = 1.0 / (resistance * capacitance)
    membrane_decay = decay(membrane_rate, dt)
    spike_decay = decay(b_spike, dt)
    voltage_decay = decay(b_voltage, dt)
    asc1_decay = decay(k_asc1, dt)
    asc2_decay = decay(k_asc2, dt)
    voltage_convolution = exponential_convolution(b_voltage, membrane_rate, dt)

    for index in 1:n_steps
        if refractory_remaining > 0.0
            refractory_remaining = max(0.0, refractory_remaining - dt)
            trace[index] = v
            continue
        end
        total_current = current + i_asc1 + i_asc2
        equilibrium_offset = resistance * total_current
        voltage_offset = v - e_l
        next_offset = equilibrium_offset + (voltage_offset - equilibrium_offset) * membrane_decay
        v = e_l + next_offset
        theta_spike *= spike_decay
        i_asc1 *= asc1_decay
        i_asc2 *= asc2_decay
        threshold_forcing = equilibrium_offset * (1.0 - voltage_decay) / b_voltage +
            (voltage_offset - equilibrium_offset) * voltage_convolution
        theta_voltage = theta_voltage * voltage_decay + a_voltage * threshold_forcing
        all(isfinite, (v, theta_spike, i_asc1, i_asc2, theta_voltage)) ||
            throw(OverflowError("GLIF5 candidate is non-finite"))
        if v > theta_inf + theta_spike + theta_voltage
            v = e_l + f_v * (v - e_l) - delta_v
            theta_spike += delta_theta_spike
            i_asc1 = f_asc1 * i_asc1 + delta_i_asc1
            i_asc2 = f_asc2 * i_asc2 + delta_i_asc2
            refractory_remaining = refractory_period
            events += 1
        end
        all(isfinite, (v, theta_spike, i_asc1, i_asc2, theta_voltage)) ||
            throw(OverflowError("GLIF5 reset is non-finite"))
        trace[index] = v
    end
    return (
        trace = trace,
        events = events,
        vf = v,
        theta_spike_f = theta_spike,
        i_asc1_f = i_asc1,
        i_asc2_f = i_asc2,
        theta_voltage_f = theta_voltage,
        refractory_f = refractory_remaining,
    )
end

end # module GLIF5Accel
