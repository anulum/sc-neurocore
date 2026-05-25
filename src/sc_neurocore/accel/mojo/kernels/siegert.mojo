# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for siegert

from std.math import exp, sqrt


fn _siegert_finite(x: Float64) -> Bool:
    var residual = x - x
    return x == x and residual == 0.0


fn siegert_valid(
    tau_m: Float64, tau_rp: Float64, v_threshold: Float64, v_reset: Float64, v_rest: Float64
) -> Bool:
    return (
        _siegert_finite(tau_m)
        and tau_m > 0.0
        and _siegert_finite(tau_rp)
        and tau_rp > 0.0
        and _siegert_finite(v_threshold)
        and _siegert_finite(v_reset)
        and _siegert_finite(v_rest)
        and v_threshold > v_reset
    )


fn _erf_approx(x: Float64) -> Float64:
    var sign = 1.0
    if x < 0.0:
        sign = -1.0
    var a = x
    if a < 0.0:
        a = -a
    var t = 1.0 / (1.0 + 0.3275911 * a)
    var poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * exp(-a * a))


fn _siegert_contrib(node: Float64, weight: Float64, half: Float64, mid: Float64) -> Float64:
    var u = half * node + mid
    var u2 = u * u
    if u2 > 50.0:
        u2 = 50.0
    return weight * exp(u2) * (1.0 + _erf_approx(u))


fn siegert_step(
    current: Float64,
    tau_m: Float64,
    tau_rp: Float64,
    v_threshold: Float64,
    v_reset: Float64,
    v_rest: Float64,
) -> Float64:
    if not _siegert_finite(current) or not siegert_valid(tau_m, tau_rp, v_threshold, v_reset, v_rest):
        return -1.0
    var mu = v_rest + current
    if not _siegert_finite(mu):
        return -1.0
    var sigma = current
    if sigma < 0.0:
        sigma = -sigma
    sigma = sigma * 0.1
    if sigma < 1.0e-6:
        sigma = 1.0e-6
    if not _siegert_finite(sigma) or sigma <= 0.0:
        return -1.0
    var u_th = (v_threshold - mu) / sigma
    var u_re = (v_reset - mu) / sigma
    if not _siegert_finite(u_th) or not _siegert_finite(u_re) or u_th <= u_re:
        return -1.0
    var half = 0.5 * (u_th - u_re)
    var mid = 0.5 * (u_th + u_re)
    if not _siegert_finite(half) or not _siegert_finite(mid) or half <= 0.0:
        return -1.0
    var integral = 0.0
    integral += _siegert_contrib(-0.993128599185095, 0.017614007139152, half, mid)
    integral += _siegert_contrib(-0.963971927277914, 0.040601429800387, half, mid)
    integral += _siegert_contrib(-0.912234428251326, 0.062672048334109, half, mid)
    integral += _siegert_contrib(-0.839116971822219, 0.083276741576704, half, mid)
    integral += _siegert_contrib(-0.746331906460151, 0.101930119817240, half, mid)
    integral += _siegert_contrib(-0.636053680726515, 0.118194531961518, half, mid)
    integral += _siegert_contrib(-0.510867001950827, 0.131688638449177, half, mid)
    integral += _siegert_contrib(-0.373706088715420, 0.142096109318382, half, mid)
    integral += _siegert_contrib(-0.227785851141645, 0.149172986472604, half, mid)
    integral += _siegert_contrib(-0.076526521133497, 0.152753387130726, half, mid)
    integral += _siegert_contrib(0.076526521133497, 0.152753387130726, half, mid)
    integral += _siegert_contrib(0.227785851141645, 0.149172986472604, half, mid)
    integral += _siegert_contrib(0.373706088715420, 0.142096109318382, half, mid)
    integral += _siegert_contrib(0.510867001950827, 0.131688638449177, half, mid)
    integral += _siegert_contrib(0.636053680726515, 0.118194531961518, half, mid)
    integral += _siegert_contrib(0.746331906460151, 0.101930119817240, half, mid)
    integral += _siegert_contrib(0.839116971822219, 0.083276741576704, half, mid)
    integral += _siegert_contrib(0.912234428251326, 0.062672048334109, half, mid)
    integral += _siegert_contrib(0.963971927277914, 0.040601429800387, half, mid)
    integral += _siegert_contrib(0.993128599185095, 0.017614007139152, half, mid)
    integral = integral * half
    if not _siegert_finite(integral) or integral < 0.0:
        return -1.0
    var t_isi = tau_rp + tau_m * sqrt(3.141592653589793) * integral
    if not _siegert_finite(t_isi) or t_isi < tau_rp:
        return -1.0
    var rate = 1000.0 / t_isi
    var max_rate = 1000.0 / tau_rp
    if not _siegert_finite(rate) or rate < 0.0 or rate > max_rate:
        return -1.0
    return rate
