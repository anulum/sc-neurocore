# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia N-step simulator for the Wilson-Cowan 1972 E/I model

"""
Batch parity with `WilsonCowanUnit.step` in
`src/sc_neurocore/neurons/models/wilson_cowan.py` (Wilson & Cowan 1972,
Biophys. J. 12:1–24).

Per step:
  s_e = sigmoid(w_ee · E − w_ei · I + ext)
  s_i = sigmoid(w_ie · E − w_ii · I)
  E += (−E + s_e) · dt / τ_e
  I += (−I + s_i) · dt / τ_i

where `sigmoid(x) = 1 / (1 + exp(−a·(x − θ)))`. Deterministic — no
noise — so bit-exact parity with the Python / Rust / Go / Mojo
backends requires only matching arithmetic.
"""
module WilsonCowanAccel

export simulate_wilson_cowan!, sigmoid_wc, validate_wc

@inline function logistic_wc(z::Real)::Float64
    zf = Float64(z)
    if zf >= 0.0
        return 1.0 / (1.0 + exp(-zf))
    end
    exp_z = exp(zf)
    return exp_z / (1.0 + exp_z)
end

@inline function finite_rate_wc(x::Real, a::Real, theta::Real)::Bool
    xf = Float64(x)
    baseline = logistic_wc(-Float64(a) * Float64(theta))
    return isfinite(xf) && -baseline <= xf <= 1.0 - baseline
end

@inline function sigmoid_wc(a::Real, theta::Real, x::Real)::Float64
    # Published Wilson-Cowan 1972 two-term sigmoid:
    #   S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    af = Float64(a)
    θ = Float64(theta)
    xf = Float64(x)
    isfinite(af) || throw(ArgumentError("a must be finite"))
    isfinite(θ) || throw(ArgumentError("theta must be finite"))
    isfinite(xf) || throw(ArgumentError("sigmoid input must be finite"))
    baseline = logistic_wc(-af * θ)
    return logistic_wc(af * (xf - θ)) - baseline
end

function validate_wc(
    e::Real,
    i::Real,
    w_ee::Real,
    w_ei::Real,
    w_ie::Real,
    w_ii::Real,
    tau_e::Real,
    tau_i::Real,
    a::Real,
    theta::Real,
    dt::Real,
)::Bool
    values = Float64.((w_ee, w_ei, w_ie, w_ii, tau_e, tau_i, a, theta, dt))
    return finite_rate_wc(e, a, theta) &&
        finite_rate_wc(i, a, theta) &&
        all(isfinite, values) &&
        all(x -> x >= 0.0, values[1:4]) &&
        values[5] > 0.0 &&
        values[6] > 0.0 &&
        values[7] > 0.0 &&
        values[9] > 0.0
end

"""
Run `length(ext_input)` Wilson-Cowan iterations into caller-allocated
output buffers. Accepts `Real` for scalars so PythonCall / juliacall
can pass `Int64` / `Float64` without dispatch-fail.

Returns `(e_final, i_final)` as a tuple.
"""
function simulate_wilson_cowan!(
    e_init::Real,
    i_init::Real,
    w_ee::Real,
    w_ei::Real,
    w_ie::Real,
    w_ii::Real,
    tau_e::Real,
    tau_i::Real,
    a::Real,
    theta::Real,
    dt::Real,
    ext_input::AbstractVector{<:Real},
    e_out::AbstractVector{<:Real},
    i_out::AbstractVector{<:Real},
)
    n = length(ext_input)
    length(e_out) == n || throw(ArgumentError("e_out length mismatch"))
    length(i_out) == n || throw(ArgumentError("i_out length mismatch"))

    e = Float64(e_init)
    i = Float64(i_init)
    wee = Float64(w_ee)
    wei = Float64(w_ei)
    wie = Float64(w_ie)
    wii = Float64(w_ii)
    τe = Float64(tau_e)
    τi = Float64(tau_i)
    af = Float64(a)
    θ = Float64(theta)
    δt = Float64(dt)
    validate_wc(e, i, wee, wei, wie, wii, τe, τi, af, θ, δt) ||
        throw(ArgumentError("invalid Wilson-Cowan numerical configuration"))

    @inbounds for t in 1:n
        isfinite(Float64(ext_input[t])) || throw(ArgumentError("external input must be finite"))
        s_e = sigmoid_wc(af, θ, wee * e - wei * i + Float64(ext_input[t]))
        s_i = sigmoid_wc(af, θ, wie * e - wii * i)
        next_e = e + (-e + s_e) / τe * δt
        next_i = i + (-i + s_i) / τi * δt
        finite_rate_wc(next_e, af, θ) && finite_rate_wc(next_i, af, θ) ||
            throw(ArgumentError("invalid Wilson-Cowan candidate state"))
        e = next_e
        i = next_i
        e_out[t] = e
        i_out[t] = i
    end
    return (e, i)
end

end # module WilsonCowanAccel
