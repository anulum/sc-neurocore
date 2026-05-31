module BoothRinzel

export BoothRinzelState, step!, reset!, simulate

mutable struct BoothRinzelState
    Vs::Float64
    Vd::Float64
    h::Float64
    n::Float64
    q::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_ca::Float64
    g_kca::Float64
    g_l::Float64
    g_c::Float64
    p::Float64
    c_m::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    alpha_ca::Float64
    k_ca::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
end

BoothRinzelState() = BoothRinzelState(
    -60.0, -60.0, 0.6, 0.1, 0.1, 0.1,
    120.0, 36.0, 2.0, 5.0, 0.3, 1.0, 0.5, 1.0,
    50.0, -77.0, 120.0, -54.4, 0.002, 0.01, 0.01, 0.01, 0.0,
)

_clip(x, lo, hi) = min(max(x, lo), hi)
_safe_exp(x) = exp(_clip(x, -100.0, 100.0))
_gate(x) = isfinite(x) && 0.0 <= x <= 1.0

function _valid_config(s::BoothRinzelState)::Bool
    positives = (s.g_na, s.g_k, s.g_ca, s.g_kca, s.g_l, s.g_c, s.c_m, s.alpha_ca, s.k_ca, s.f_ca, s.dt)
    all(x -> isfinite(x) && x > 0.0, positives) || return false
    isfinite(s.p) && 0.0 < s.p < 1.0 || return false
    all(isfinite, (s.e_na, s.e_k, s.e_ca, s.e_l, s.v_threshold))
end

function _valid_state(Vs, Vd, h, n, q, ca)::Bool
    isfinite(Vs) && isfinite(Vd) && isfinite(ca) && ca >= 0.0 || return false
    _gate(h) && _gate(n) && _gate(q) || return false
    -200.0 <= Vs <= 100.0 && -200.0 <= Vd <= 100.0
end

function _substep(s::BoothRinzelState, Vs, Vd, h, n, q, ca, I_ext, dt)
    isfinite(I_ext) && isfinite(dt) && dt > 0.0 || return nothing
    m_inf = 1.0 / (1.0 + _safe_exp(-(Vs + 30.0) / 9.5))
    h_inf = 1.0 / (1.0 + _safe_exp((Vs + 53.0) / 7.0))
    n_inf = 1.0 / (1.0 + _safe_exp(-(Vs + 30.0) / 10.0))
    q_inf = 1.0 / (1.0 + _safe_exp(-(Vd + 25.0) / 5.0))

    tau_h = 1.0 + 7.0 / (_safe_exp((Vs + 40.0) / 5.0) + _safe_exp(-(Vs + 40.0) / 5.0))
    tau_n = 1.0 + 5.0 / (_safe_exp((Vs + 35.0) / 10.0) + _safe_exp(-(Vs + 35.0) / 10.0))
    tau_q = 10.0

    h = _clip(h + dt * (h_inf - h) / tau_h, 0.0, 1.0)
    n = _clip(n + dt * (n_inf - n) / tau_n, 0.0, 1.0)
    q = _clip(q + dt * (q_inf - q) / tau_q, 0.0, 1.0)

    i_na = s.g_na * m_inf^3 * h * (Vs - s.e_na)
    i_k = s.g_k * n^4 * (Vs - s.e_k)
    i_l = s.g_l * (Vs - s.e_l)
    i_c = s.g_c * (Vs - Vd)
    i_ca = s.g_ca * q^2 * (Vd - s.e_ca)
    i_kca = s.g_kca * (ca / (ca + s.k_ca)) * (Vd - s.e_k)

    dVs = (I_ext - i_na - i_k - i_l - i_c) / (s.c_m * s.p)
    dVd = (-i_ca - i_kca - i_l + i_c) / (s.c_m * (1.0 - s.p))
    dCa = -s.alpha_ca * i_ca - s.f_ca * ca

    Vs = _clip(Vs + dt * dVs, -200.0, 100.0)
    Vd = _clip(Vd + dt * dVd, -200.0, 100.0)
    ca = max(0.0, ca + dt * dCa)
    _valid_state(Vs, Vd, h, n, q, ca) ? (Vs, Vd, h, n, q, ca) : nothing
end

function step!(s::BoothRinzelState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    _valid_config(s) && isfinite(I_ext) && _valid_state(s.Vs, s.Vd, s.h, s.n, s.q, s.ca) || return -1
    old_Vs = s.Vs
    Vs, Vd, h, n, q, ca = s.Vs, s.Vd, s.h, s.n, s.q, s.ca
    sub_dt = dt / 4.0
    for _ in 1:4
        next_state = _substep(s, Vs, Vd, h, n, q, ca, I_ext, sub_dt)
        next_state === nothing && return -1
        Vs, Vd, h, n, q, ca = next_state
    end
    s.Vs, s.Vd, s.h, s.n, s.q, s.ca = Vs, Vd, h, n, q, ca
    old_Vs < s.v_threshold && s.Vs >= s.v_threshold ? 1 : 0
end

function reset!(s::BoothRinzelState)
    s.Vs = -60.0
    s.Vd = -60.0
    s.h = 0.6
    s.n = 0.1
    s.q = 0.1
    s.ca = 0.1
    return s
end

function simulate(s::BoothRinzelState, I_ext::Float64, steps::Int; dt::Float64=s.dt)
    trace = Float64[]
    steps <= 0 && return trace
    sizehint!(trace, steps)
    for _ in 1:steps
        step!(s, I_ext; dt=dt)
        push!(trace, s.Vs)
    end
    trace
end

end
