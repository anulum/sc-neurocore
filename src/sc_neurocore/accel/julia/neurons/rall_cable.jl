# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for rall_cable

module RallCableAccel

export step!, simulate, RallCableNeuronState

mutable struct RallCableNeuronState
    n_comp::Float64
    tau_m::Float64
    v_rest::Float64
    g_ratio::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
    v::Float64
end

function RallCableNeuronState()
    RallCableNeuronState(5.0, 20.0, -65.0, 0.5, -50.0, -65.0, 0.1, 0.0)
end

function step!(s::RallCableNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev_soma = s.v[0]
        dv = zeros(s.n_comp)
        for i in 1:s.n_comp
            leak = -(s.v[i] - s.v_rest)
            left = (i > 0) ? s.v[i - 1] : s.v[i]
            right = (i < s.n_comp - 1) ? s.v[i + 1] : s.v[i]
            axial = s.g_ratio * (left - 2.0 * s.v[i] + right)
            inj = (i == s.n_comp - 1) ? I_ext : 0.0
            dv[i] = (leak + axial + inj) / s.tau_m
        end
        s.v += dv * s.dt
        if s.v[0] >= s.v_threshold && v_prev_soma < s.v_threshold
            s.v[0] = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = RallCableNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.n_comp
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module RallCableAccel
