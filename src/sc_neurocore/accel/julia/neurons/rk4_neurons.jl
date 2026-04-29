# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 neuron parity kernels

"""
Maintained Julia RK4 batch kernels for the first three explicit
integrator-path neuron models.

These functions mirror the Python RK4 implementations in:

- `neurons/sc_izhikevich.py`
- `neurons/models/hodgkin_huxley.py`
- `neurons/models/adex.py`

They intentionally use fixed-step RK4 arithmetic instead of Julia's
adaptive ODE solvers so parity tests can compare trajectories directly.
"""
module Rk4NeuronsAccel

export simulate_izhikevich_rk4!, simulate_hodgkin_huxley_rk4!, simulate_adex_rk4!

const IZH_SPIKE_THRESHOLD = 30.0

@inline function _izh_rhs(v::Float64, u::Float64, current::Float64)
    dv = 0.04 * v^2 + 5.0 * v + 140.0 - u + current
    du = 0.02 * (0.2 * v - u)
    return dv, du
end

function simulate_izhikevich_rk4!(
    currents::AbstractVector{<:Real},
    dt::Real,
    v_out::AbstractVector{<:Real},
    u_out::AbstractVector{<:Real},
    spikes_out::AbstractVector{UInt64},
)
    n = length(currents)
    length(v_out) == n || throw(ArgumentError("v_out length mismatch"))
    length(u_out) == n || throw(ArgumentError("u_out length mismatch"))
    length(spikes_out) >= n || throw(ArgumentError("spikes_out length mismatch"))

    δt = Float64(dt)
    v = -65.0
    u = 0.2 * v
    n_spikes = 0

    @inbounds for idx in 1:n
        current = Float64(currents[idx])
        k1v, k1u = _izh_rhs(v, u, current)
        k2v, k2u = _izh_rhs(v + 0.5 * δt * k1v, u + 0.5 * δt * k1u, current)
        k3v, k3u = _izh_rhs(v + 0.5 * δt * k2v, u + 0.5 * δt * k2u, current)
        k4v, k4u = _izh_rhs(v + δt * k3v, u + δt * k3u, current)

        v += (δt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u += (δt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)

        if v >= IZH_SPIKE_THRESHOLD
            v = -65.0
            u += 8.0
            n_spikes += 1
            spikes_out[n_spikes] = UInt64(idx - 1)
        end

        v_out[idx] = v
        u_out[idx] = u
    end
    return n_spikes
end

@inline function _adex_rhs(v::Float64, w::Float64, current::Float64)
    exp_arg = clamp((v - (-55.0)) / 2.0, -20.0, 20.0)
    exp_term = 2.0 * exp(exp_arg)
    dv = (-(v - (-65.0)) + exp_term) / 20.0 + (-w + current) / 200.0
    dw = (0.5 * (v - (-65.0)) - w) / 100.0
    return dv, dw
end

function simulate_adex_rk4!(
    currents::AbstractVector{<:Real},
    dt::Real,
    v_out::AbstractVector{<:Real},
    w_out::AbstractVector{<:Real},
    spikes_out::AbstractVector{UInt64},
)
    n = length(currents)
    length(v_out) == n || throw(ArgumentError("v_out length mismatch"))
    length(w_out) == n || throw(ArgumentError("w_out length mismatch"))
    length(spikes_out) >= n || throw(ArgumentError("spikes_out length mismatch"))

    δt = Float64(dt)
    v = -65.0
    w = 0.0
    n_spikes = 0

    @inbounds for idx in 1:n
        current = Float64(currents[idx])
        k1v, k1w = _adex_rhs(v, w, current)
        k2v, k2w = _adex_rhs(v + 0.5 * δt * k1v, w + 0.5 * δt * k1w, current)
        k3v, k3w = _adex_rhs(v + 0.5 * δt * k2v, w + 0.5 * δt * k2w, current)
        k4v, k4w = _adex_rhs(v + δt * k3v, w + δt * k3w, current)

        v += (δt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        w += (δt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

        if v >= -50.0
            v = -68.0
            w += 7.0
            n_spikes += 1
            spikes_out[n_spikes] = UInt64(idx - 1)
        end

        v_out[idx] = v
        w_out[idx] = w
    end
    return n_spikes
end

@inline function _alpha_m(v::Float64)
    d = v + 40.0
    if abs(d) < 1e-7
        return 1.0
    end
    return 0.1 * d / (1.0 - exp(-d / 10.0))
end

@inline _beta_m(v::Float64) = 4.0 * exp(-(v + 65.0) / 18.0)
@inline _alpha_h(v::Float64) = 0.07 * exp(-(v + 65.0) / 20.0)
@inline _beta_h(v::Float64) = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))

@inline function _alpha_n(v::Float64)
    d = v + 55.0
    if abs(d) < 1e-7
        return 0.1
    end
    return 0.01 * d / (1.0 - exp(-d / 10.0))
end

@inline _beta_n(v::Float64) = 0.125 * exp(-(v + 65.0) / 80.0)

@inline function _hh_rhs(v::Float64, m::Float64, h::Float64, n::Float64, current::Float64)
    am = _alpha_m(v)
    bm = _beta_m(v)
    ah = _alpha_h(v)
    bh = _beta_h(v)
    an = _alpha_n(v)
    bn = _beta_n(v)

    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    i_na = 120.0 * m^3 * h * (v - 50.0)
    i_k = 36.0 * n^4 * (v - (-77.0))
    i_l = 0.3 * (v - (-54.4))
    dv = -i_na - i_k - i_l + current
    return dv, dm, dh, dn
end

function simulate_hodgkin_huxley_rk4!(
    currents::AbstractVector{<:Real},
    dt::Real,
    v_out::AbstractVector{<:Real},
    m_out::AbstractVector{<:Real},
    h_out::AbstractVector{<:Real},
    n_out::AbstractVector{<:Real},
    spikes_out::AbstractVector{UInt64},
)
    n_steps = length(currents)
    length(v_out) == n_steps || throw(ArgumentError("v_out length mismatch"))
    length(m_out) == n_steps || throw(ArgumentError("m_out length mismatch"))
    length(h_out) == n_steps || throw(ArgumentError("h_out length mismatch"))
    length(n_out) == n_steps || throw(ArgumentError("n_out length mismatch"))
    length(spikes_out) >= n_steps || throw(ArgumentError("spikes_out length mismatch"))

    δt = Float64(dt)
    substeps = round(Int, 1.0 / δt)
    v = -65.0
    m = 0.05
    h = 0.6
    n = 0.32
    n_spikes = 0

    @inbounds for idx in 1:n_steps
        v_prev = v
        current = Float64(currents[idx])
        for _ in 1:substeps
            k1v, k1m, k1h, k1n = _hh_rhs(v, m, h, n, current)
            k2v, k2m, k2h, k2n = _hh_rhs(
                v + 0.5 * δt * k1v,
                m + 0.5 * δt * k1m,
                h + 0.5 * δt * k1h,
                n + 0.5 * δt * k1n,
                current,
            )
            k3v, k3m, k3h, k3n = _hh_rhs(
                v + 0.5 * δt * k2v,
                m + 0.5 * δt * k2m,
                h + 0.5 * δt * k2h,
                n + 0.5 * δt * k2n,
                current,
            )
            k4v, k4m, k4h, k4n = _hh_rhs(
                v + δt * k3v,
                m + δt * k3m,
                h + δt * k3h,
                n + δt * k3n,
                current,
            )

            v += (δt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            m += (δt / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
            h += (δt / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
            n += (δt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
        end

        if v >= 0.0 && v_prev < 0.0
            n_spikes += 1
            spikes_out[n_spikes] = UInt64(idx - 1)
        end

        v_out[idx] = v
        m_out[idx] = m
        h_out[idx] = h
        n_out[idx] = n
    end
    return n_spikes
end

end # module Rk4NeuronsAccel
