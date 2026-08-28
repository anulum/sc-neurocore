# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Wilson 1999 polynomial cortical model (parity with wilson_hr.py)

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron.simulate` bit-for-bit. The
# right-hand side is exact polynomial arithmetic (no transcendental functions), so
# an identical RK4 operation order yields an identical continuous `v` trace,
# upward-crossing count, and final `(v, r)` state.
#
# Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.

module WilsonHRAccel

export simulate_trace

function simulate_trace(
    v0::Float64,
    r0::Float64,
    capacitance::Float64,
    tau_r::Float64,
    v_peak::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0 || !all(
        isfinite,
        (v0, r0, capacitance, tau_r, v_peak, dt, current),
    )
        throw(ArgumentError("Wilson-HR batch inputs must be finite and n_steps non-negative"))
    end
    if capacitance <= 0.0 || tau_r <= 0.0 || dt <= 0.0
        throw(ArgumentError("Wilson-HR capacitance, tau_r, and dt must be positive"))
    end
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    r = r0
    deriv(vv, rr) = (
        (-(17.81 + 47.71 * vv + 32.63 * vv * vv) * (vv - 0.55) - 26.0 * rr * (vv + 0.92) + current) / capacitance,
        (-rr + 1.35 * vv + 1.03) / tau_r,
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        v_previous = v
        dv1, dr1 = deriv(v, r)
        dv2, dr2 = deriv(v + 0.5 * dt * dv1, r + 0.5 * dt * dr1)
        dv3, dr3 = deriv(v + 0.5 * dt * dv2, r + 0.5 * dt * dr2)
        dv4, dr4 = deriv(v + dt * dv3, r + dt * dr3)
        stages = (dv1, dr1, dv2, dr2, dv3, dr3, dv4, dr4)
        if !all(isfinite, stages)
            throw(DomainError(stages, "Wilson-HR RK4 stage became non-finite"))
        end
        next_v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        next_r = r + dt * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4) / 6.0
        if !all(isfinite, (next_v, next_r))
            throw(DomainError((next_v, next_r), "Wilson-HR candidate became non-finite"))
        end
        v, r = next_v, next_r
        if v >= v_peak && v_previous < v_peak
            spikes += 1
        end
        trace[t] = v
    end
    return (trace = trace, spikes = spikes, vf = v, rf = r)
end

end # module WilsonHRAccel
