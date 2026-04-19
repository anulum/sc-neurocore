# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia N-step simulator for Wong-Wang 2006 decision unit

"""
Batch parity with `WongWangUnit.step` in
`src/sc_neurocore/neurons/models/wong_wang.py` (Wong & Wang 2006,
J. Neurosci. 26:1314–1328).

Per step:
  1. iₖ = j_n · sₖ − j_cross · s₍₃₋ₖ₎ + i₀ + stimₖ + σ · ξ
  2. rₖ = φ(iₖ) where φ(i) = (a·i − b) / (1 − exp(−d·(a·i − b)))
     with singularity guard `|a·i − b| < 1e-6 → 1/d`.
  3. sₖ += (−sₖ/τₛ + (1 − sₖ) · γ · rₖ) · dt
  4. clamp sₖ into [0, 1].

Caller passes pre-drawn `xi` of length `2 * length(stim1)` so the
trajectory is bit-exact with the Python `numpy.random.randn()` order
(matches the Rust + PINGCircuit pattern: Python owns the RNG).
"""
module WongWangAccel

export simulate_wong_wang!, phi_wong_wang

const A = 270.0
const B = 108.0
const D = 0.154

@inline function phi_wong_wang(i_syn::Real)::Float64
    x = A * i_syn - B
    if abs(x) < 1e-6
        return 1.0 / D
    end
    return x / (1.0 - exp(-D * x))
end

"""
Run `length(stim1)` Wong-Wang iterations into caller-allocated output
buffers. Accepts `Real` for every scalar so PythonCall / juliacall can
pass `Int64` / `Float64` without dispatch-fail.

Returns `(s1_final, s2_final)` as a tuple.
"""
function simulate_wong_wang!(
    s1_init::Real,
    s2_init::Real,
    tau_s::Real,
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
    r1_out::AbstractVector{<:Real},
    r2_out::AbstractVector{<:Real},
)
    n = length(stim1)
    length(stim2) == n || throw(ArgumentError("stim1 and stim2 length mismatch"))
    length(xi) == 2 * n || throw(ArgumentError("xi length must be 2 * n_steps"))
    length(s1_out) == n || throw(ArgumentError("s1_out length mismatch"))
    length(s2_out) == n || throw(ArgumentError("s2_out length mismatch"))
    length(r1_out) == n || throw(ArgumentError("r1_out length mismatch"))
    length(r2_out) == n || throw(ArgumentError("r2_out length mismatch"))

    s1 = Float64(s1_init)
    s2 = Float64(s2_init)
    τs = Float64(tau_s)
    γ = Float64(gamma)
    jn = Float64(j_n)
    jx = Float64(j_cross)
    i0 = Float64(i_0)
    σ = Float64(sigma)
    δt = Float64(dt)

    @inbounds for t in 1:n
        xi1 = Float64(xi[2 * t - 1])
        xi2 = Float64(xi[2 * t])
        i1 = jn * s1 - jx * s2 + i0 + Float64(stim1[t]) + σ * xi1
        i2 = jn * s2 - jx * s1 + i0 + Float64(stim2[t]) + σ * xi2
        r1 = phi_wong_wang(i1)
        r2 = phi_wong_wang(i2)
        s1 += (-s1 / τs + (1.0 - s1) * γ * r1) * δt
        s2 += (-s2 / τs + (1.0 - s2) * γ * r2) * δt
        s1 = clamp(s1, 0.0, 1.0)
        s2 = clamp(s2, 0.0, 1.0)
        s1_out[t] = s1
        s2_out[t] = s2
        r1_out[t] = r1
        r2_out[t] = r2
    end
    return (s1, s2)
end

end # module WongWangAccel
