# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for dpi_neuron

module DpiNeuronAccel

export step!, simulate, DPINeuronState

mutable struct DPINeuronState
    i_mem::Float64
    i_threshold::Float64
    i_reset::Float64
    i_leak::Float64
    tau::Float64
    gain::Float64
    dt::Float64
end

function DPINeuronState()
    DPINeuronState(0.0, 1.0, 0.0, 0.01, 20.0, 1.0, 1.0)
end

function step!(s::DPINeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        di = (-s.i_mem + s.gain * i_syn + s.i_leak) / s.tau * s.dt
        s.i_mem += di
        s.i_mem = max(s.i_mem, 0.0)
        if s.i_mem >= s.i_threshold
            s.i_mem = s.i_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DPINeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.i_mem
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DpiNeuronAccel
