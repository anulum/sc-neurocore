# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for expif

module ExpifAccel

export step!, simulate, ExpIFNeuronState

mutable struct ExpIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    v_rh::Float64
    delta_t::Float64
    tau::Float64
    dt::Float64
end

function ExpIFNeuronState()
    ExpIFNeuronState(-65.0, -65.0, -68.0, -50.0, -55.0, 2.0, 20.0, 0.1)
end

function _rhs(s::ExpIFNeuronState, v::Float64, I_ext::Float64)
    exp_term = s.delta_t * exp(clamp((v - s.v_rh) / s.delta_t, -20.0, 20.0))
    rhs = (-(v - s.v_rest) + exp_term + I_ext) / s.tau
    if !all(isfinite, (exp_term, rhs))
        throw(DomainError(rhs, "ExpIF RK4 derivative must remain finite"))
    end
    return rhs
end

function step!(s::ExpIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    s.dt = dt
    if !all(isfinite, (s.v, s.v_rest, s.v_reset, s.v_threshold, s.v_rh, s.delta_t, s.tau, s.dt))
        throw(DomainError(s.v, "ExpIF state parameters must be finite"))
    end
    if s.delta_t <= 0.0 || s.tau <= 0.0 || s.dt <= 0.0
        throw(DomainError(s.delta_t, "ExpIF delta_t, tau, and dt must be positive"))
    end
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "ExpIF input current must be finite"))
    end

    k1 = _rhs(s, s.v, I_ext)
    k2 = _rhs(s, s.v + 0.5 * s.dt * k1, I_ext)
    k3 = _rhs(s, s.v + 0.5 * s.dt * k2, I_ext)
    k4 = _rhs(s, s.v + s.dt * k3, I_ext)
    next_v = s.v + s.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if !isfinite(next_v)
        throw(DomainError(next_v, "ExpIF RK4 update must remain finite"))
    end

    s.v = next_v
    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ExpIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ExpifAccel
