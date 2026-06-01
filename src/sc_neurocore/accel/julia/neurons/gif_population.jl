# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gif_population

module GifPopulationAccel

export step!, simulate, reset!, GIFPopulationNeuronState

mutable struct GIFPopulationNeuronState
    v::Float64
    theta::Float64
    eta::Float64
    tau_m::Float64
    tau_eta::Float64
    delta_v::Float64
    lambda_0::Float64
    eta_increment::Float64
    v_rest::Float64
    v_reset::Float64
    dt::Float64
    seed::UInt64
    rng::UInt64
end

function GIFPopulationNeuronState(seed::UInt64=UInt64(42))
    normalized_seed = seed == 0 ? UInt64(1) : seed
    GIFPopulationNeuronState(-65.0, -50.0, 0.0, 20.0, 100.0, 2.0, 0.001, 5.0, -65.0, -65.0, 0.5, normalized_seed, normalized_seed)
end

function finite_values(values::Float64...)
    all(isfinite, values)
end

function valid_runtime(s::GIFPopulationNeuronState)
    finite_values(s.v, s.theta, s.eta, s.tau_m, s.tau_eta, s.delta_v, s.lambda_0, s.eta_increment, s.v_rest, s.v_reset, s.dt) &&
        s.tau_m > 0.0 && s.tau_eta > 0.0 && s.delta_v > 0.0 && s.lambda_0 >= 0.0 && s.dt > 0.0
end

function uniform!(s::GIFPopulationNeuronState)
    x = s.rng
    x ⊻= x >> 12
    x ⊻= x << 25
    x ⊻= x >> 27
    s.rng = x
    Float64((x * UInt64(2685821657736338717)) >> 11) * (1.0 / 9007199254740992.0)
end

function advance_subthreshold(s::GIFPopulationNeuronState, I_ext::Float64)
    eta_decay = exp(-s.dt / s.tau_eta)
    membrane_decay = exp(-s.dt / s.tau_m)
    x0 = s.v - s.v_rest - I_ext
    eta_new = s.eta * eta_decay
    if abs(s.tau_m - s.tau_eta) <= 1e-12
        x_new = membrane_decay * (x0 - s.eta * s.dt / s.tau_m)
    else
        coupling = s.tau_eta / (s.tau_eta - s.tau_m)
        x_new = x0 * membrane_decay - s.eta * coupling * (eta_decay - membrane_decay)
    end
    v_new = s.v_rest + I_ext + x_new
    return v_new, eta_new, finite_values(v_new, eta_new)
end

function spike_probability(s::GIFPopulationNeuronState, voltage::Float64)
    s.lambda_0 == 0.0 && return 0.0
    exponent = clamp((voltage - s.theta) / s.delta_v, -745.0, 20.0)
    hazard = s.lambda_0 * exp(exponent)
    clamp(1.0 - exp(-hazard * s.dt), 0.0, 1.0)
end

function step!(s::GIFPopulationNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    s.dt = dt
    if !isfinite(I_ext) || !valid_runtime(s)
        return 0
    end
    v_candidate, eta_candidate, ok = advance_subthreshold(s, I_ext)
    if !ok
        return 0
    end
    s.v = v_candidate
    s.eta = eta_candidate
    if uniform!(s) < spike_probability(s, s.v)
        s.v = s.v_reset
        s.eta += s.eta_increment
        return 1
    end
    return 0
end

function reset!(s::GIFPopulationNeuronState)
    s.v = s.v_rest
    s.eta = 0.0
    s.rng = s.seed
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    s = GIFPopulationNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module GifPopulationAccel
