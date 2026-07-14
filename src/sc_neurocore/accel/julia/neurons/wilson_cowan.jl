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
  dE/dt = (−E + sigmoid(w_ee · E − w_ei · I + ext)) / τ_e
  dI/dt = (−I + sigmoid(w_ie · E − w_ii · I)) / τ_i
  (E, I) advance through one fixed-step RK4 update.

where `sigmoid(x) = logistic(a·(x − θ)) − logistic(−a·θ)`. This is the
maintained normalised reduction: the original availability/refractory factors
and independent inhibitory external drive are outside its declared scope. No
noise is present; the public bounded floating-point trajectory contract keeps
the RK4 arithmetic order aligned while allowing small libm differences.
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
    return isfinite(xf) && -baseline <= xf <= 1.0
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

@inline function wilson_cowan_derivatives(
    e::Float64,
    i::Float64,
    ext::Float64,
    wee::Float64,
    wei::Float64,
    wie::Float64,
    wii::Float64,
    τe::Float64,
    τi::Float64,
    af::Float64,
    θ::Float64,
)::Tuple{Float64,Float64}
    s_e = sigmoid_wc(af, θ, wee * e - wei * i + ext)
    s_i = sigmoid_wc(af, θ, wie * e - wii * i)
    return ((-e + s_e) / τe, (-i + s_i) / τi)
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
    next_e_out = Vector{Float64}(undef, n)
    next_i_out = Vector{Float64}(undef, n)

    @inbounds for t in 1:n
        ext = Float64(ext_input[t])
        isfinite(ext) || throw(ArgumentError("external input must be finite"))
        k1_e, k1_i = wilson_cowan_derivatives(e, i, ext, wee, wei, wie, wii, τe, τi, af, θ)
        k2_e, k2_i = wilson_cowan_derivatives(
            e + 0.5 * δt * k1_e,
            i + 0.5 * δt * k1_i,
            ext,
            wee,
            wei,
            wie,
            wii,
            τe,
            τi,
            af,
            θ,
        )
        k3_e, k3_i = wilson_cowan_derivatives(
            e + 0.5 * δt * k2_e,
            i + 0.5 * δt * k2_i,
            ext,
            wee,
            wei,
            wie,
            wii,
            τe,
            τi,
            af,
            θ,
        )
        k4_e, k4_i = wilson_cowan_derivatives(
            e + δt * k3_e,
            i + δt * k3_i,
            ext,
            wee,
            wei,
            wie,
            wii,
            τe,
            τi,
            af,
            θ,
        )
        next_e = e + δt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0
        next_i = i + δt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0
        finite_rate_wc(next_e, af, θ) && finite_rate_wc(next_i, af, θ) ||
            throw(ArgumentError("invalid Wilson-Cowan candidate state"))
        e = next_e
        i = next_i
        next_e_out[t] = e
        next_i_out[t] = i
    end
    copyto!(e_out, next_e_out)
    copyto!(i_out, next_i_out)
    return (e, i)
end

end # module WilsonCowanAccel
