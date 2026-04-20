# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for spinnaker_lif

module SpinnakerLifAccel

export step!, simulate, SpiNNakerLIFNeuronState

mutable struct SpiNNakerLIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    i_offset::Float64
    tau_refrac::Float64
    refrac_count::Float64
    dt::Float64
end

function SpiNNakerLIFNeuronState()
    SpiNNakerLIFNeuronState(-70.0, -70.0, -70.0, -50.0, 20.0, 0.0, 2.0, 0.0, 1.0)
end

function step!(s::SpiNNakerLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if s.refrac_count > 0
            s.refrac_count -= s.dt
            return 0
        end
        s.v += (-(s.v - s.v_rest) + (I_ext + s.i_offset)) / s.tau_m * s.dt
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.refrac_count = s.tau_refrac
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SpiNNakerLIFNeuronState()
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

end # module SpinnakerLifAccel
