# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Medvedev 2005 first-return map

# Source contract: calibrated Section-4 slow-calcium recurrence from Medvedev,
# Physica D 202 (2005), 37-59, DOI 10.1016/j.physd.2005.01.021. The source map
# is recovered at current=0; current is a maintained active-return perturbation.

module MedvedevMapAccel

export simulate_trace

function _valid_parameters(
    beta_0::Float64,
    beta_hc::Float64,
    beta_sn::Float64,
    delta::Float64,
    decay_t0::Float64,
    alpha_t0::Float64,
    f_0::Float64,
    f_1::Float64,
    homoclinic_exponent::Float64,
    d::Float64,
    input_gain::Float64,
)
    values = (
        beta_0,
        beta_hc,
        beta_sn,
        delta,
        decay_t0,
        alpha_t0,
        f_0,
        f_1,
        homoclinic_exponent,
        d,
        input_gain,
    )
    return all(isfinite, values) &&
           0.0 < beta_0 < beta_sn < beta_hc < delta &&
           0.0 < decay_t0 < 1.0 &&
           0.0 < alpha_t0 < 1.0 &&
           0.0 <= f_1 < f_0 &&
           homoclinic_exponent > 0.0 &&
           d > 0.0 &&
           input_gain >= 0.0
end

"""
    simulate_trace(u0, beta_0, beta_hc, beta_sn, delta, decay_t0,
                   alpha_t0, f_0, f_1, homoclinic_exponent, d, input_gain,
                   n_steps, current)

Run the checked Medvedev slow-calcium first-return map. Return
`(trace, events, uf)`, where events count pre-step states `u <= u_HC`.
"""
function simulate_trace(
    u0::Float64,
    beta_0::Float64,
    beta_hc::Float64,
    beta_sn::Float64,
    delta::Float64,
    decay_t0::Float64,
    alpha_t0::Float64,
    f_0::Float64,
    f_1::Float64,
    homoclinic_exponent::Float64,
    d::Float64,
    input_gain::Float64,
    n_steps::Int,
    current::Float64,
)
    if n_steps < 0 || !isfinite(u0) || !isfinite(current) ||
       !_valid_parameters(
        beta_0,
        beta_hc,
        beta_sn,
        delta,
        decay_t0,
        alpha_t0,
        f_0,
        f_1,
        homoclinic_exponent,
        d,
        input_gain,
    )
        throw(ArgumentError("invalid Medvedev first-return request"))
    end

    u_0 = beta_0 / (delta - beta_0)
    u_hc = beta_hc / (delta - beta_hc)
    u_sn = beta_sn / (delta - beta_sn)
    trace = Vector{Float64}(undef, n_steps)
    u = u0
    events = 0
    for index in 1:n_steps
        events += u <= u_hc
        candidate = if u <= u_0
            decay_t0 * u + (1.0 - decay_t0) * f_0 + input_gain * current
        elseif u <= u_hc
            u_1 = (1.0 - alpha_t0) * u + alpha_t0 * f_0
            gap = beta_hc - delta * u_1 / (1.0 + u_1)
            inner_return = if gap <= 0.0
                f_1
            else
                log_argument = d * gap
                if !isfinite(log_argument) || log_argument <= 0.0
                    throw(DomainError(log_argument, "invalid Medvedev log argument"))
                end
                scale = exp(homoclinic_exponent * log(log_argument))
                scale * (u_1 - f_1) + f_1
            end
            inner_return + input_gain * current
        else
            u_sn
        end
        if !isfinite(candidate)
            throw(DomainError(candidate, "non-finite Medvedev candidate"))
        end
        u = candidate
        trace[index] = u
    end
    return (trace = trace, events = events, uf = u)
end

end # module MedvedevMapAccel
