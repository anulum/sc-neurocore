# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for siegert

module SiegertAccel

export step!, simulate, SiegertTransferFunctionState

mutable struct SiegertTransferFunctionState
    tau_m::Float64
    tau_rp::Float64
    v_threshold::Float64
    v_reset::Float64
    v_rest::Float64
end

function SiegertTransferFunctionState()
    SiegertTransferFunctionState(20.0, 2.0, -50.0, -70.0, -65.0)
end

function step!(s::SiegertTransferFunctionState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        mu = s.v_rest + I_ext
        sigma = max(abs(I_ext) * 0.1, 1e-06)
        u_th = (s.v_threshold - mu) / sigma
        u_re = (s.v_reset - mu) / sigma
        n_quad = 40
        (u_pts, w_pts) = np.polynomial.legendre.leggauss(n_quad)
        half_range = 0.5 * (u_th - u_re)
        mid = 0.5 * (u_th + u_re)
        u_scaled = half_range * u_pts + mid
        integrand = exp(clamp(u_scaled ^ 2, nothing, 50.0)) * (1.0 + _erf_approx(u_scaled))
        integral_val = Float64(half_range * sum(w_pts * integrand))
        t_isi = s.tau_rp + s.tau_m * sqrt(pi) * integral_val
        return 1000.0 / max(t_isi, 0.01)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SiegertTransferFunctionState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.tau_m
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SiegertAccel
