# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for prescott

module PrescottAccel

export step!, simulate, PrescottNeuronState

mutable struct PrescottNeuronState
    v::Float64
    w::Float64
    g_fast::Float64
    g_slow::Float64
    g_l::Float64
    e_fast::Float64
    e_slow::Float64
    e_l::Float64
    beta_w::Float64
    gamma_w::Float64
    tau_w::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function PrescottNeuronState()
    PrescottNeuronState(-65.0, 0.0, 20.0, 20.0, 2.0, 50.0, -100.0, -70.0, -21.0, 15.0, 100.0, 0.15, 0.1, -20.0)
end

function step!(s::PrescottNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_inf = 1.0 / (1.0 + exp(-(s.v + 20.0) / 15.0))
        w_inf = 1.0 / (1.0 + exp(-(s.v - s.beta_w) / s.gamma_w))
        i_fast = s.g_fast * m_inf * (s.v - s.e_fast)
        i_slow = s.g_slow * s.w * (s.v - s.e_slow)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_fast - i_slow - i_l + I_ext) * s.dt
        s.w += s.phi * (w_inf - s.w) / s.tau_w * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PrescottNeuronState()
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

end # module PrescottAccel
