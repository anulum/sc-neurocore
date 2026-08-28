# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Terman-Wang 1995 LEGION relaxation oscillator (parity with terman_wang.py)

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.terman_wang.TermanWangOscillator.simulate`. The
# cubic is exact (`v*v*v`, matching the engine `v.powi(3)`); the `tanh` gating uses
# Julia's libm, so focused parity tests bound the complete trace and require
# identical spike counts on the enrolled operating regimes.
#
# Reference: Terman, D. & Wang, D.L. (1995). Physica D 81:148-176.

module TermanWangAccel

export simulate_trace

"""
    simulate_trace(v0, w0, alpha, beta, eps, rho, dt, v_peak, n_steps, current)

Run a failure-atomic Terman-Wang RK4 batch. Invalid inputs, stages, or
candidates raise before a result is returned.
"""
function simulate_trace(
    v0::Float64,
    w0::Float64,
    alpha::Float64,
    beta::Float64,
    eps::Float64,
    rho::Float64,
    dt::Float64,
    v_peak::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0 || !all(isfinite, (v0, w0, alpha, beta, eps, rho, dt, v_peak, current))
        throw(ArgumentError("Terman-Wang batch inputs must be finite and n_steps non-negative"))
    end
    if beta <= 0.0 || eps <= 0.0 || dt <= 0.0
        throw(ArgumentError("Terman-Wang beta, epsilon, and dt must be positive"))
    end
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    w = w0
    deriv(vv, ww) = (
        3.0 * vv - vv * vv * vv + 2.0 - ww + current + rho,
        eps * (alpha * (1.0 + tanh(vv / beta)) - ww),
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        v_prev = v
        dv1, dw1 = deriv(v, w)
        dv2, dw2 = deriv(v + 0.5 * dt * dv1, w + 0.5 * dt * dw1)
        dv3, dw3 = deriv(v + 0.5 * dt * dv2, w + 0.5 * dt * dw2)
        dv4, dw4 = deriv(v + dt * dv3, w + dt * dw3)
        stages = (dv1, dw1, dv2, dw2, dv3, dw3, dv4, dw4)
        if !all(isfinite, stages)
            throw(DomainError(stages, "Terman-Wang RK4 stage became non-finite"))
        end
        next_v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        next_w = w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0
        if !all(isfinite, (next_v, next_w))
            throw(DomainError((next_v, next_w), "Terman-Wang candidate became non-finite"))
        end
        v, w = next_v, next_w
        trace[t] = v
        if v >= v_peak && v_prev < v_peak
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w)
end

end # module TermanWangAccel
