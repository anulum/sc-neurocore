# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for e_prop_alif

module EPropAlifAccel

export step!, simulate, EPropALIFNeuronState

mutable struct EPropALIFNeuronState
    v::Float64
    a::Float64
    e_trace::Float64
    tau_m::Float64
    tau_a::Float64
    v_threshold_base::Float64
    beta::Float64
    v_reset::Float64
    dt::Float64
    alpha_m::Float64
    alpha_a::Float64
end

function EPropALIFNeuronState()
    EPropALIFNeuronState(0.0, 0.0, 0.0, 20.0, 200.0, 1.0, 0.07, 0.0, 1.0, 0.0, 0.0)
end

function step!(s::EPropALIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v = s.alpha_m * s.v + I_ext
        threshold = s.v_threshold_base + s.beta * s.a
        psi = max(0.0, 1.0 - abs(s.v - threshold)) * 0.3
        s.e_trace = s.alpha_a * s.e_trace + psi
        if s.v >= threshold
            s.v = s.v_reset
            s.a = s.alpha_a * s.a + 1.0
            return 1
        end
        s.a *= s.alpha_a
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EPropALIFNeuronState()
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

end # module EPropAlifAccel
