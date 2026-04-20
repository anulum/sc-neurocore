# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for glm_neuron

module GlmNeuronAccel

export step!, simulate, GLMNeuronState

mutable struct GLMNeuronState
    n_k::Float64
    n_h::Float64
    mu::Float64
    dt_ms::Float64
    k::Float64
    h::Float64
    _stim_buf::Float64
    _spike_buf::Float64
    _rng::Float64
end

function GLMNeuronState()
    GLMNeuronState(10.0, 20.0, -3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step!(s::GLMNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s._stim_buf = (s._stim_buf, 1)
        s._stim_buf[0] = stimulus
        log_rate = Float64((s.k, s._stim_buf) + (s.h, s._spike_buf) + s.mu)
        lam = exp(clamp(log_rate, -20.0, 20.0))
        p = lam * s.dt_ms / 1000.0
        spike = (s._rng.random() < min(p, 1.0)) ? 1 : 0
        s._spike_buf = (s._spike_buf, 1)
        s._spike_buf[0] = Float64(spike)
        return spike
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GLMNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.n_k
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module GlmNeuronAccel
