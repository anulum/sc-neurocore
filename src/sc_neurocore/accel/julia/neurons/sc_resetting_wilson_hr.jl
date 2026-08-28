# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia retained resetting Wilson-HR project recurrence

"""
Retained unit-capacitance Wilson-HR project recurrence with hard reset.

This module deliberately has a distinct SC identity and makes no claim that
the reset rule is part of Wilson's continuous 1999 source equations.
"""
module SCResettingWilsonHRAccel

export simulate_trace

"""
    simulate_trace(v0, r0, tau_r, v_peak, dt, n_steps, current)

Run a failure-atomic constant-current batch. The result contains the complete
post-step voltage trace, event count, and final dynamic state. Invalid inputs
or any non-finite RK4 stage raise without returning a candidate state.
"""
function simulate_trace(
    v0::Float64,
    r0::Float64,
    tau_r::Float64,
    v_peak::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0 || !all(isfinite, (v0, r0, tau_r, v_peak, dt, current))
        throw(ArgumentError("SC resetting Wilson-HR inputs must be finite and n_steps non-negative"))
    end
    if tau_r <= 0.0 || dt <= 0.0
        throw(ArgumentError("SC resetting Wilson-HR tau_r and dt must be positive"))
    end
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    r = r0
    derivatives(voltage, recovery) = (
        -(17.81 + 47.71 * voltage + 32.63 * voltage * voltage) * (voltage - 0.55) -
        26.0 * recovery * (voltage + 0.92) + current,
        (-recovery + 1.35 * voltage + 1.03) / tau_r,
    )
    events = 0
    @inbounds for index in 1:n_steps
        dv1, dr1 = derivatives(v, r)
        dv2, dr2 = derivatives(v + 0.5 * dt * dv1, r + 0.5 * dt * dr1)
        dv3, dr3 = derivatives(v + 0.5 * dt * dv2, r + 0.5 * dt * dr2)
        dv4, dr4 = derivatives(v + dt * dv3, r + dt * dr3)
        stages = (dv1, dr1, dv2, dr2, dv3, dr3, dv4, dr4)
        if !all(isfinite, stages)
            throw(DomainError(stages, "SC resetting Wilson-HR RK4 stage became non-finite"))
        end
        next_v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        next_r = r + dt * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4) / 6.0
        if !all(isfinite, (next_v, next_r))
            throw(DomainError((next_v, next_r), "SC resetting Wilson-HR candidate became non-finite"))
        end
        if next_v >= v_peak
            next_v = -0.7
            events += 1
        end
        v, r = next_v, next_r
        trace[index] = v
    end
    return (trace = trace, spikes = events, vf = v, rf = r)
end

end # module SCResettingWilsonHRAccel
