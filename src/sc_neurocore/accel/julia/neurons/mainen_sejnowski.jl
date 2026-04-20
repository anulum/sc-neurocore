# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mainen_sejnowski

module MainenSejnowskiAccel

export step!, simulate, MainenSejnowskiNeuronState

mutable struct MainenSejnowskiNeuronState
    vs::Float64
    va::Float64
    m::Float64
    h::Float64
    n::Float64
    kappa::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_s::Float64
    c_a::Float64
    dt::Float64
    v_threshold::Float64
end

function MainenSejnowskiNeuronState()
    MainenSejnowskiNeuronState(-65.0, -65.0, 0.05, 0.6, 0.3, 10.0, 3000.0, 1500.0, 1.0, 50.0, -90.0, -70.0, 1.0, 0.1, 0.005, -20.0)
end

function step!(s::MainenSejnowskiNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        vs_prev = s.vs
        for _ in 1:20
            am = 0.182 * (s.va + 25.0) / (1.0 - _safe_exp(-(s.va + 25.0) / 9.0) + 1e-12)
            bm = -0.124 * (s.va + 25.0) / (1.0 - _safe_exp((s.va + 25.0) / 9.0) + 1e-12)
            ah = 0.024 * (s.va + 40.0) / (1.0 - _safe_exp(-(s.va + 40.0) / 5.0) + 1e-12)
            bh = -0.0091 * (s.va + 65.0) / (1.0 - _safe_exp((s.va + 65.0) / 5.0) + 1e-12)
            an = 0.02 * (s.va - 20.0) / (1.0 - _safe_exp(-(s.va - 20.0) / 9.0) + 1e-12)
            bn = -0.002 * (s.va - 20.0) / (1.0 - _safe_exp((s.va - 20.0) / 9.0) + 1e-12)
            s.m = clamp(s.m + (am * (1 - s.m) - bm * s.m) * s.dt, 0.0, 1.0)
            s.h = clamp(s.h + (ah * (1 - s.h) - bh * s.h) * s.dt, 0.0, 1.0)
            s.n = clamp(s.n + (an * (1 - s.n) - bn * s.n) * s.dt, 0.0, 1.0)
            i_na = s.g_na * s.m ^ 3 * s.h * (s.va - s.e_na)
            i_k = s.g_k * s.n * (s.va - s.e_k)
            i_l = s.g_l * (s.vs - s.e_l)
            dvs = (-i_l + s.kappa * (s.va - s.vs) + I_ext) / s.c_s * s.dt
            dva = (-i_na - i_k + s.kappa * (s.vs - s.va)) / s.c_a * s.dt
            s.vs = Float64(clamp(s.vs + dvs, -200.0, 200.0))
            s.va = Float64(clamp(s.va + dva, -200.0, 200.0))
        end
        return (s.vs >= s.v_threshold && vs_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MainenSejnowskiNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.vs
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module MainenSejnowskiAccel
