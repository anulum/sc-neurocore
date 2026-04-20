# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for tc_lif

module TcLifAccel

export step!, simulate, TwoCompartmentLIFNeuronState

mutable struct TwoCompartmentLIFNeuronState
    v_s::Float64
    v_d::Float64
    v_rest::Float64
    v_reset::Float64
    theta::Float64
    tau_s::Float64
    tau_d::Float64
    kappa::Float64
    dt::Float64
end

function TwoCompartmentLIFNeuronState()
    TwoCompartmentLIFNeuronState(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 20.0, 0.5, 1.0)
end

function step!(s::TwoCompartmentLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dvd = (-(s.v_d - s.v_rest) + i_dend) / s.tau_d * s.dt
        s.v_d += dvd
        dvs = (-(s.v_s - s.v_rest) + s.kappa * (s.v_d - s.v_s) + i_soma) / s.tau_s * s.dt
        s.v_s += dvs
        if s.v_s >= s.theta
            s.v_s = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = TwoCompartmentLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module TcLifAccel
