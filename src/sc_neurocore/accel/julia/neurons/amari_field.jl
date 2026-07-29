# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of the Amari 1977 periodic neural field

"""Source-level Heaviside periodic-grid specialization of Amari equation (3)."""
module AmariFieldAccel

export simulate_amari_field!, validate_amari_field

"""Return whether a complete Amari numerical configuration is in domain."""
function validate_amari_field(
    u::AbstractVector{<:Real}, tau::Real, a_exc::Real, a_width::Real,
    b_inh::Real, b_width::Real, dx::Real, dt::Real,
)::Bool
    return length(u) >= 2 && all(isfinite, u) &&
        all(isfinite, (tau, a_exc, a_width, b_inh, b_width, dx, dt)) &&
        tau > 0 && a_exc >= 0 && a_width > 0 && b_inh >= 0 &&
        b_width > 0 && dx > 0 && dt > 0
end

"""
Advance a caller-owned drive matrix through the Amari field.

Rows of ``currents`` are simultaneous spatial stimuli. ``states_out`` has the
same shape and ``rates_out`` receives active-site fractions. The returned
vector is the final state. Validation and candidate failures throw before a
partial candidate is committed.
"""
function simulate_amari_field!(
    u_init::AbstractVector{<:Real}, tau::Real, a_exc::Real, a_width::Real,
    b_inh::Real, b_width::Real, dx::Real, dt::Real,
    currents::AbstractMatrix{<:Real}, states_out::AbstractMatrix{<:Real},
    rates_out::AbstractVector{<:Real},
)
    validate_amari_field(u_init, tau, a_exc, a_width, b_inh, b_width, dx, dt) ||
        throw(ArgumentError("invalid Amari field configuration"))
    steps, n = size(currents)
    length(u_init) == n || throw(ArgumentError("Amari state/input width mismatch"))
    size(states_out) == (steps, n) || throw(ArgumentError("Amari states_out shape mismatch"))
    length(rates_out) == steps || throw(ArgumentError("Amari rates_out length mismatch"))
    all(isfinite, currents) || throw(ArgumentError("Amari inputs must be finite"))
    u = Float64.(u_init)
    kernel = Vector{Float64}(undef, n)
    for offset in 0:n-1
        distance = min(offset, n - offset) * Float64(dx)
        kernel[offset + 1] = Float64(a_exc) * exp(-Float64(a_width) * distance) -
            Float64(b_inh) * exp(-Float64(b_width) * distance)
    end
    kernel[1] > 0 && kernel[div(n, 2) + 1] < 0 ||
        throw(ArgumentError("Amari kernel is not lateral inhibitory"))
    candidate = similar(u)
    scale = Float64(dt) / Float64(tau)
    @inbounds for step in 1:steps
        for i in 1:n
            convolution = 0.0
            for j in 1:n
                if u[j] > 0.0
                    offset = mod((i - 1) - (j - 1), n)
                    convolution += kernel[offset + 1]
                end
            end
            candidate[i] = u[i] + (-u[i] + convolution * Float64(dx) + Float64(currents[step, i])) * scale
        end
        all(isfinite, candidate) || throw(ArgumentError("invalid Amari candidate state"))
        u .= candidate
        states_out[step, :] .= u
        rates_out[step] = count(>(0.0), u) / n
    end
    return u
end

end # module AmariFieldAccel
