# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for siegert

module SiegertAccel

export step!, simulate, validate_siegert, SiegertTransferFunctionState

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
    if !validate_siegert(s) || !isfinite(I_ext)
        throw(DomainError((s.tau_m, s.tau_rp, I_ext), "Siegert state/current must be finite and physically ordered"))
    end

    mu = s.v_rest + I_ext
    if !isfinite(mu)
        throw(DomainError(mu, "Siegert mean voltage must remain finite"))
    end
    sigma = max(abs(I_ext) * 0.1, 1e-06)
    if !isfinite(sigma) || sigma <= 0.0
        throw(DomainError(sigma, "Siegert diffusion scale must remain finite and positive"))
    end
    u_th = (s.v_threshold - mu) / sigma
    u_re = (s.v_reset - mu) / sigma
    if !isfinite(u_th) || !isfinite(u_re) || u_th <= u_re
        throw(DomainError((u_re, u_th), "Siegert first-passage bounds must remain finite and ordered"))
    end
    n_quad = 40
    (u_pts, w_pts) = legendre_nodes_weights(n_quad)
    half_range = 0.5 * (u_th - u_re)
    mid = 0.5 * (u_th + u_re)
    if !isfinite(half_range) || !isfinite(mid) || half_range <= 0.0
        throw(DomainError((half_range, mid), "Siegert quadrature interval must remain finite"))
    end
    integral_val = 0.0
    for i in eachindex(u_pts)
        u_scaled = half_range * u_pts[i] + mid
        integrand = exp(clamp(u_scaled ^ 2, -Inf, 50.0)) * (1.0 + erf_approx(u_scaled))
        if !isfinite(integrand)
            throw(DomainError(integrand, "Siegert integrand must remain finite"))
        end
        integral_val += w_pts[i] * integrand
    end
    integral_val *= half_range
    if !isfinite(integral_val) || integral_val < 0.0
        throw(DomainError(integral_val, "Siegert integral must remain finite and non-negative"))
    end
    t_isi = s.tau_rp + s.tau_m * sqrt(pi) * integral_val
    if !isfinite(t_isi) || t_isi < s.tau_rp
        throw(DomainError(t_isi, "Siegert inter-spike interval must remain finite and refractory bounded"))
    end
    rate = 1000.0 / t_isi
    max_rate = 1000.0 / s.tau_rp
    if !isfinite(rate) || rate < 0.0 || rate > max_rate
        throw(DomainError(rate, "Siegert rate must remain finite and refractory bounded"))
    end
    return rate
end

function validate_siegert(s::SiegertTransferFunctionState)
    return isfinite(s.tau_m) && s.tau_m > 0.0 && isfinite(s.tau_rp) && s.tau_rp > 0.0 &&
           isfinite(s.v_threshold) && isfinite(s.v_reset) && isfinite(s.v_rest) &&
           s.v_threshold > s.v_reset
end

function erf_approx(x::Float64)
    sign_x = sign(x)
    a = abs(x)
    p = 0.3275911
    t = 1.0 / (1.0 + p * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign_x * (1.0 - poly * exp(-a * a))
end

function legendre_nodes_weights(n::Int)
    if n != 40
        error("only 40-point quadrature is supported")
    end
    nodes = [-0.9982377097105593, -0.990726238699457, -0.9772599499837743, -0.9579168192137917, -0.9328128082786765, -0.9020988069688743, -0.8659595032122595, -0.8246122308333117, -0.7783056514265194, -0.7273182551899271, -0.6719566846141796, -0.6125538896679802, -0.5494671250951282, -0.4830758016861787, -0.413779204371605, -0.3419940908257585, -0.2681521850072537, -0.1926975807013711, -0.1160840706752552, -0.03877241750605082, 0.03877241750605082, 0.1160840706752552, 0.1926975807013711, 0.2681521850072537, 0.3419940908257585, 0.413779204371605, 0.4830758016861787, 0.5494671250951282, 0.6125538896679802, 0.6719566846141796, 0.7273182551899271, 0.7783056514265194, 0.8246122308333117, 0.8659595032122595, 0.9020988069688743, 0.9328128082786765, 0.9579168192137917, 0.9772599499837743, 0.990726238699457, 0.9982377097105593]
    weights = [0.004521277098533191, 0.010498284531152813, 0.01642105838190789, 0.022245849194166957, 0.027937006980023402, 0.03346019528254785, 0.03878216797447202, 0.04387090818567327, 0.04869580763507223, 0.053227846983936824, 0.05743976909939155, 0.06130624249292894, 0.06480401345660104, 0.0679120458152339, 0.07061164739128678, 0.07288658239580406, 0.07472316905796826, 0.07611036190062624, 0.07703981816424797, 0.07750594797842481, 0.07750594797842481, 0.07703981816424797, 0.07611036190062624, 0.07472316905796826, 0.07288658239580406, 0.07061164739128678, 0.0679120458152339, 0.06480401345660104, 0.06130624249292894, 0.05743976909939155, 0.053227846983936824, 0.04869580763507223, 0.04387090818567327, 0.03878216797447202, 0.03346019528254785, 0.027937006980023402, 0.022245849194166957, 0.01642105838190789, 0.010498284531152813, 0.004521277098533191]
    return nodes, weights
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SiegertTransferFunctionState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = result
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SiegertAccel
