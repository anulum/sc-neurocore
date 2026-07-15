# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia batch mirror for Wong-Wang 2006

"""Publication-faithful explicit-Euler NMDA and AMPA OU recurrence."""
module WongWangAccel

export simulate_wong_wang!, phi_wong_wang, validate_wong_wang

const A = 270.0
const B = 108.0
const D = 0.154

@inline function phi_wong_wang(i_syn::Real)::Float64
    current = Float64(i_syn)
    isfinite(current) || throw(ArgumentError("synaptic current must be finite"))
    x = A * current - B
    scaled = -D * x
    response = if scaled > 700.0
        0.0
    elseif abs(x) < 1.0e-7
        1.0 / D
    else
        x / -expm1(scaled)
    end
    isfinite(response) || throw(ArgumentError("invalid Wong-Wang transfer response"))
    return max(0.0, response)
end

@inline finite_gate(value::Real)::Bool = isfinite(Float64(value)) && 0.0 <= Float64(value) <= 1.0

function validate_wong_wang(
    s1::Real,
    s2::Real,
    noise1::Real,
    noise2::Real,
    tau_s::Real,
    tau_ampa::Real,
    gamma::Real,
    j_n::Real,
    j_cross::Real,
    i_0::Real,
    sigma::Real,
    dt::Real,
)::Bool
    values = Float64.((noise1, noise2, tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt))
    return finite_gate(s1) && finite_gate(s2) && all(isfinite, values) &&
        values[3] > 0.0 && values[4] > 0.0 && values[5] > 0.0 &&
        values[6] >= 0.0 && values[7] >= 0.0 && values[9] >= 0.0 && values[10] > 0.0
end

"""
Advance a deterministic-sample Wong-Wang batch into caller-owned buffers.

The returned tuple contains final ``s1``, ``s2``, ``noise1``, and ``noise2``.
"""
function simulate_wong_wang!(
    s1_init::Real,
    s2_init::Real,
    noise1_init::Real,
    noise2_init::Real,
    tau_s::Real,
    tau_ampa::Real,
    gamma::Real,
    j_n::Real,
    j_cross::Real,
    i_0::Real,
    sigma::Real,
    dt::Real,
    stim1::AbstractVector{<:Real},
    stim2::AbstractVector{<:Real},
    xi::AbstractVector{<:Real},
    s1_out::AbstractVector{<:Real},
    s2_out::AbstractVector{<:Real},
    noise1_out::AbstractVector{<:Real},
    noise2_out::AbstractVector{<:Real},
    r1_out::AbstractVector{<:Real},
    r2_out::AbstractVector{<:Real},
)
    steps = length(stim1)
    length(stim2) == steps || throw(ArgumentError("stim1 and stim2 length mismatch"))
    length(xi) == 2 * steps || throw(ArgumentError("xi length must be 2 * n_steps"))
    for (name, output) in (
        ("s1", s1_out),
        ("s2", s2_out),
        ("noise1", noise1_out),
        ("noise2", noise2_out),
        ("r1", r1_out),
        ("r2", r2_out),
    )
        length(output) == steps || throw(ArgumentError("$(name)_out length mismatch"))
    end

    s1 = Float64(s1_init)
    s2 = Float64(s2_init)
    noise1 = Float64(noise1_init)
    noise2 = Float64(noise2_init)
    tau = Float64(tau_s)
    tau_noise = Float64(tau_ampa)
    gm = Float64(gamma)
    jn = Float64(j_n)
    jx = Float64(j_cross)
    background = Float64(i_0)
    noise_sigma = Float64(sigma)
    step_size = Float64(dt)
    validate_wong_wang(
        s1, s2, noise1, noise2, tau, tau_noise, gm, jn, jx, background, noise_sigma, step_size
    ) || throw(ArgumentError("invalid Wong-Wang numerical configuration"))
    noise_scale = sqrt(step_size / tau_noise) * noise_sigma
    noise_decay = step_size / tau_noise

    @inbounds for step in 1:steps
        drive1 = Float64(stim1[step])
        drive2 = Float64(stim2[step])
        xi1 = Float64(xi[2 * step - 1])
        xi2 = Float64(xi[2 * step])
        all(isfinite, (drive1, drive2, xi1, xi2)) ||
            throw(ArgumentError("stimuli and Gaussian samples must be finite"))
        rate1 = phi_wong_wang(jn * s1 - jx * s2 + background + drive1 + noise1)
        rate2 = phi_wong_wang(jn * s2 - jx * s1 + background + drive2 + noise2)
        next_s1 = s1 + step_size * (-s1 / tau + (1.0 - s1) * gm * rate1)
        next_s2 = s2 + step_size * (-s2 / tau + (1.0 - s2) * gm * rate2)
        next_noise1 = noise1 - noise_decay * noise1 + noise_scale * xi1
        next_noise2 = noise2 - noise_decay * noise2 + noise_scale * xi2
        finite_gate(next_s1) && finite_gate(next_s2) &&
            isfinite(next_noise1) && isfinite(next_noise2) ||
            throw(ArgumentError("invalid Wong-Wang candidate state"))
        s1, s2 = next_s1, next_s2
        noise1, noise2 = next_noise1, next_noise2
        s1_out[step] = s1
        s2_out[step] = s2
        noise1_out[step] = noise1
        noise2_out[step] = noise2
        r1_out[step] = rate1
        r2_out[step] = rate2
    end
    return (s1, s2, noise1, noise2)
end

end # module WongWangAccel
