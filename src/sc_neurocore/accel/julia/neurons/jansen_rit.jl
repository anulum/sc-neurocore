# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia batch mirror for Jansen–Rit 1995

"""Equation-(6) explicit-Euler recurrence for one cortical column."""
module JansenRitAccel

export simulate_jansen_rit!, sigmoid_jansen_rit, validate_jansen_rit

@inline function sigmoid_jansen_rit(voltage::Real, e0::Real, v0::Real, slope::Real)::Float64
    value = Float64(voltage)
    all(isfinite, (value, e0, v0, slope)) ||
        throw(ArgumentError("Jansen–Rit sigmoid values must be finite"))
    exponent = Float64(slope) * (Float64(v0) - value)
    if exponent >= 0.0
        exp_neg = exp(-exponent)
        return 2.0 * Float64(e0) * exp_neg / (1.0 + exp_neg)
    end
    return 2.0 * Float64(e0) / (1.0 + exp(exponent))
end

function validate_jansen_rit(values::NTuple{15, Float64})::Bool
    return all(isfinite, values) &&
        values[7] > 0.0 && values[8] > 0.0 &&
        values[9] > 0.0 && values[10] > 0.0 && values[11] >= 0.0 &&
        values[12] > 0.0 && values[14] > 0.0 && values[15] > 0.0
end

"""
Advance a Jansen–Rit drive batch into seven caller-owned trace buffers.

The returned tuple contains final ``y0``, ``y3``, ``y1``, ``y4``, ``y2``,
and ``y5`` in the public state order.
"""
function simulate_jansen_rit!(
    y0_init::Real,
    y3_init::Real,
    y1_init::Real,
    y4_init::Real,
    y2_init::Real,
    y5_init::Real,
    a_exc::Real,
    b_exc::Real,
    a_rate::Real,
    b_rate::Real,
    c::Real,
    e0::Real,
    v0::Real,
    slope::Real,
    dt::Real,
    p_ext::AbstractVector{<:Real},
    y0_out::AbstractVector{<:Real},
    y3_out::AbstractVector{<:Real},
    y1_out::AbstractVector{<:Real},
    y4_out::AbstractVector{<:Real},
    y2_out::AbstractVector{<:Real},
    y5_out::AbstractVector{<:Real},
    eeg_out::AbstractVector{<:Real},
)
    steps = length(p_ext)
    for (name, output) in (
        ("y0", y0_out), ("y3", y3_out), ("y1", y1_out),
        ("y4", y4_out), ("y2", y2_out), ("y5", y5_out), ("eeg", eeg_out),
    )
        length(output) == steps || throw(ArgumentError("$(name)_out length mismatch"))
    end
    values = Float64.((
        y0_init, y3_init, y1_init, y4_init, y2_init, y5_init,
        a_exc, b_exc, a_rate, b_rate, c, e0, v0, slope, dt,
    ))
    configuration = Tuple(values)
    validate_jansen_rit(configuration) ||
        throw(ArgumentError("invalid Jansen–Rit numerical configuration"))
    all(isfinite, p_ext) || throw(ArgumentError("p_ext must contain only finite values"))

    y0, y3, y1, y4, y2, y5 = values[1:6]
    gain_a, gain_b = values[7], values[8]
    rate_a, rate_b, c1 = values[9], values[10], values[11]
    e0f, v0f, rf, step_size = values[12], values[13], values[14], values[15]
    c2, c3, c4 = 0.8 * c1, 0.25 * c1, 0.25 * c1
    @inbounds for step in 1:steps
        s_pyramidal = sigmoid_jansen_rit(y1 - y2, e0f, v0f, rf)
        s_excitatory = sigmoid_jansen_rit(c1 * y0, e0f, v0f, rf)
        s_inhibitory = sigmoid_jansen_rit(c3 * y0, e0f, v0f, rf)
        candidate = (
            y0 + step_size * y3,
            y3 + step_size * (gain_a * rate_a * s_pyramidal - 2.0 * rate_a * y3 - rate_a^2 * y0),
            y1 + step_size * y4,
            y4 + step_size * (gain_a * rate_a * (Float64(p_ext[step]) + c2 * s_excitatory) - 2.0 * rate_a * y4 - rate_a^2 * y1),
            y2 + step_size * y5,
            y5 + step_size * (gain_b * rate_b * c4 * s_inhibitory - 2.0 * rate_b * y5 - rate_b^2 * y2),
        )
        all(isfinite, candidate) || throw(ArgumentError("invalid Jansen–Rit candidate state"))
        y0, y3, y1, y4, y2, y5 = candidate
        y0_out[step], y3_out[step] = y0, y3
        y1_out[step], y4_out[step] = y1, y4
        y2_out[step], y5_out[step] = y2, y5
        eeg_out[step] = y1 - y2
    end
    return (y0, y3, y1, y4, y2, y5)
end

end # module JansenRitAccel
