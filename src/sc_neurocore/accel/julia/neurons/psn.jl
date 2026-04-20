# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for psn

module PsnAccel

export step!, simulate, ParallelSpikingNeuronState

mutable struct ParallelSpikingNeuronState
    kernel_size::Float64
    v_threshold::Float64
    kernel::Float64
    buffer::Float64
    _ptr::Float64
end

function ParallelSpikingNeuronState()
    ParallelSpikingNeuronState(8.0, 1.0, 0.0, 0.0, 0.0)
end

function step!(s::ParallelSpikingNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.buffer[s._ptr % s.kernel_size] = I_ext
        s._ptr += 1
        n = min(s._ptr, s.kernel_size)
        score = Float64((s.kernel[:n], s.buffer[:n]))
        if score >= s.v_threshold
            s.buffer[:] = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ParallelSpikingNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.kernel_size
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PsnAccel
