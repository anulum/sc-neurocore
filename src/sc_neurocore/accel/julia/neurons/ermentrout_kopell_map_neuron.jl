# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Ermentrout-Kopell theta map

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.ermentrout_kopell_map_neuron.ErmentroutKopellMapNeuron.simulate`.
# The only transcendental is `cos`, and the theta neuron is a non-chaotic phase
# oscillator, so Julia's `cos` (which may differ from the reference libm by a
# ULP) does not amplify: the trace stays within a small ULP band and the spike
# counts match. The wrap uses `mod(theta, 2*pi)` (floored remainder), matching
# Python's `theta % (2*pi)`.
#
# Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233-253.

module ErmentroutKopellMapAccel

export simulate_trace

"""
    simulate_trace(theta0, dt, gain, theta_threshold, n_steps, current)

Run `n_steps` of the Ermentrout-Kopell theta map from phase `theta0` under a
constant input `current`. Returns a named tuple `(trace, spikes, thetaf)` where
`trace[t]` is `theta` after step `t` (wrapped to [0, 2*pi)), `spikes` counts
upward crossings of `theta_threshold`, and `thetaf` is the final phase.
"""
function simulate_trace(
    theta0::Float64,
    dt::Float64,
    gain::Float64,
    theta_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    all(isfinite, (theta0, dt, gain, theta_threshold, current)) ||
        throw(ArgumentError("Ermentrout-Kopell inputs must be finite"))
    dt > 0.0 || throw(ArgumentError("dt must be positive"))
    trace = Vector{Float64}(undef, n_steps)
    theta = theta0
    inp = gain * current
    two_pi = 2.0 * pi
    spikes = 0
    for t in 1:n_steps
        theta_prev = theta
        cos_theta = cos(theta)
        d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp
        theta_next = theta + dt * d_theta
        isfinite(d_theta) && isfinite(theta_next) ||
            throw(OverflowError("Ermentrout-Kopell candidate phase became non-finite"))
        if theta_next >= theta_threshold && theta_prev < theta_threshold
            spikes += 1
        end
        theta = mod(theta_next, two_pi)
        trace[t] = theta
    end
    return (trace = trace, spikes = spikes, thetaf = theta)
end

end # module ErmentroutKopellMapAccel
