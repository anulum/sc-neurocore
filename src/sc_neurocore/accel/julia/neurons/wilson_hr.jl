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
# an identical RK4 operation order yields an identical `v` trace (already hard-reset
# to -0.7 on spiking steps), spike count, and final `(v, r)` state.
#
# Reference: Wilson, H.R. (1999). J. Theor. Biol. 200:375-388.

module WilsonHRAccel

export simulate_trace

function simulate_trace(
    v0::Float64,
    r0::Float64,
    tau_r::Float64,
    v_peak::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    r = r0
    deriv(vv, rr) = (
        -(17.81 + 47.71 * vv + 32.63 * vv * vv) * (vv - 0.55) - 26.0 * rr * (vv + 0.92) + current,
        (-rr + 1.35 * vv + 1.03) / tau_r,
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        dv1, dr1 = deriv(v, r)
        dv2, dr2 = deriv(v + 0.5 * dt * dv1, r + 0.5 * dt * dr1)
        dv3, dr3 = deriv(v + 0.5 * dt * dv2, r + 0.5 * dt * dr2)
        dv4, dr4 = deriv(v + dt * dv3, r + dt * dr3)
        v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        r = r + dt * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4) / 6.0
        if v >= v_peak
            v = -0.7
            spikes += 1
        end
        trace[t] = v
    end
    return (trace = trace, spikes = spikes, vf = v, rf = r)
end

end # module WilsonHRAccel
