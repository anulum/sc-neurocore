# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/bcm

module BcmAccel

using Statistics, LinearAlgebra

mutable struct BCMSynapseState
    eta::Float64
    tau_theta::Float64
    theta_init::Float64
    w_min::Float64
    w_max::Float64
    weight::Float64
end

function BCMSynapseState()
    BCMSynapseState(0.01, 1000.0, 0.1, 0.0, 1.0, 0.5)
end

function step(s::BCMSynapseState, pre_rate, post_rate, dt)
    # BCM update: dw = eta * y * (y - theta_M) * x
    dw = s.eta * post_rate * (post_rate - s.theta_m) * pre_rate * dt
    s.weight += dw
    s.weight = max(s.w_min, min(s.w_max, s.weight))
    # Sliding threshold: d(theta)/dt = (y^2 - theta) / tau_theta
    s.theta_m += (post_rate^2 - s.theta_m) * dt / s.tau_theta
    return s.weight
end

function reset(s::BCMSynapseState)
    s.theta_m = s.theta_init
end

end # module BcmAccel
