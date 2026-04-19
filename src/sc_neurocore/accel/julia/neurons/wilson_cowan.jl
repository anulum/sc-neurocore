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

export simulate_wilson_cowan!, sigmoid_wc

@inline function sigmoid_wc(a::Real, theta::Real, x::Real)::Float64
    # Published Wilson-Cowan 1972 two-term sigmoid:
    #   S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    baseline = 1.0 / (1.0 + exp(Float64(a) * Float64(theta)))
    return 1.0 / (1.0 + exp(-Float64(a) * (Float64(x) - Float64(theta)))) - baseline
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

    @inbounds for t in 1:n
        s_e = sigmoid_wc(af, θ, wee * e - wei * i + Float64(ext_input[t]))
        s_i = sigmoid_wc(af, θ, wie * e - wii * i)
        e += (-e + s_e) / τe * δt
        i += (-i + s_i) / τi * δt
        e_out[t] = e
        i_out[t] = i
    end
    return (e, i)
end

end # module WilsonCowanAccel
