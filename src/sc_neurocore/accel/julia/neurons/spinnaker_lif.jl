# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for spinnaker_lif

module SpinnakerLifAccel

export step!, simulate, valid, SpiNNakerLIFNeuronState

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

function step!(s::SpiNNakerLIFNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !valid(s) || !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0
        return -1
    end
    s.dt = dt
    if s.refrac_count > 0.0
        s.refrac_count = max(0.0, s.refrac_count - s.dt)
        return 0
    end
    steady = s.v_rest + I_ext + s.i_offset
    next_v = steady + (s.v - steady) * exp(-s.dt / s.tau_m)
    if !isfinite(next_v)
        return -1
    end
    if next_v >= s.v_threshold
        s.v = s.v_reset
        s.refrac_count = s.tau_refrac
        return 1
    end
    s.v = next_v
    return 0
end

function valid(s::SpiNNakerLIFNeuronState)
    return isfinite(s.v) &&
           isfinite(s.v_rest) &&
           isfinite(s.v_reset) &&
           isfinite(s.v_threshold) &&
           s.v_threshold > s.v_reset &&
           isfinite(s.tau_m) &&
           s.tau_m > 0.0 &&
           isfinite(s.i_offset) &&
           isfinite(s.tau_refrac) &&
           s.tau_refrac >= 0.0 &&
           isfinite(s.refrac_count) &&
           s.refrac_count >= 0.0 &&
           isfinite(s.dt) &&
           s.dt > 0.0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
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
