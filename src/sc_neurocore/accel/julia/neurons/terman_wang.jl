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
# Julia's libm, so the trace is within a per-step ULP band of the NumPy reference
# (the two-dimensional relaxation oscillator is non-chaotic, so it does not
# amplify) with identical spike counts.
#
# Reference: Terman, D. & Wang, D.L. (1995). Physica D 81:148-176.

module TermanWangAccel

export simulate_trace

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
        v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        w = w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0
        trace[t] = v
        if v >= v_peak && v_prev < v_peak
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w)
end

end # module TermanWangAccel
