# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for benda_herz

module BendaHerzAccel

export step!, simulate, BendaHerzNeuronState

mutable struct BendaHerzNeuronState
    a::Float64
    f_max::Float64
    beta::Float64
    i_half::Float64
    tau_a::Float64
    delta_a::Float64
    dt::Float64
    _rng::Float64
end

function BendaHerzNeuronState()
    BendaHerzNeuronState(0.0, 200.0, 0.1, 5.0, 100.0, 0.5, 1.0, 0.0)
end

function _f_onset(s::BendaHerzNeuronState, x)
    return s.f_max / (1.0 + exp(-s.beta * (x - s.i_half)))
end

function step!(s::BendaHerzNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        rate = s._f_onset(I_ext - s.a)
        s.a += (-s.a / s.tau_a + s.delta_a * rate) * s.dt
        p = rate * s.dt / 1000.0
        return (s._rng.random() < min(p, 1.0)) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BendaHerzNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.a
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module BendaHerzAccel
