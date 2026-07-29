# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Neuron Model Zoo (31 models)

module NeuronZoo

using Printf
using Statistics

# ============================================================
# §1  AdEx (Adaptive Exponential IF)
# ============================================================
function adex_step(V, w, I; C=281.0, gL=30.0, EL=-70.6, ΔT=2.0, VT=-50.4, τw=144.0, a=4.0, dt=0.1)
    dV = (-gL*(V-EL) + gL*ΔT*exp((V-VT)/ΔT) - w + I) / C
    dw = (a*(V-EL) - w) / τw
    V_new = V + dV*dt
    w_new = w + dw*dt
    spike = V_new >= 20.0
    if spike; V_new = EL; w_new += 80.5; end
    return V_new, w_new, spike
end

# §2  Brette-Gerstner AdEx variant
function brainscales_adex_step(V, w, I; C=200.0, gL=10.0, EL=-65.0, ΔT=1.5, VT=-52.0, τw=120.0, a=2.0, dt=0.1)
    return adex_step(V, w, I; C=C, gL=gL, EL=EL, ΔT=ΔT, VT=VT, τw=τw, a=a, dt=dt)
end

# §3  ExpIF (Exponential IF)
function expif_step(V, I; τm=20.0, EL=-65.0, ΔT=2.0, VT=-50.0, Vreset=-65.0, dt=0.1)
    dV = (-(V-EL) + ΔT*exp((V-VT)/ΔT) + I) / τm
    V_new = V + dV*dt
    spike = V_new >= 0.0
    if spike; V_new = Vreset; end
    return V_new, spike
end

# §4  COBA-LIF (conductance-based)
function coba_lif_step(V, ge, gi, I; τm=20.0, τe=5.0, τi=10.0, EL=-60.0, Ee=0.0, Ei=-80.0, Vth=-50.0, Vr=-60.0, dt=0.1)
    dge = -ge/τe
    dgi = -gi/τi
    dV = (-(V-EL) - ge*(V-Ee) - gi*(V-Ei) + I) / τm
    V_new = V + dV*dt
    ge_new = ge + dge*dt
    gi_new = gi + dgi*dt
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, ge_new, gi_new, spike
end

# §5  GLIF (Generalized LIF)
function glif_step(V, θ, I; τm=20.0, τθ=100.0, EL=-65.0, Vr=-65.0, θ0=-50.0, θ_step=1.0, dt=0.1)
    dV = (-(V-EL) + I) / τm
    dθ = -(θ - θ0) / τθ
    V_new = V + dV*dt
    θ_new = θ + dθ*dt
    spike = V_new >= θ_new
    if spike; V_new = Vr; θ_new += θ_step; end
    return V_new, θ_new, spike
end

# §6  Wilson-Cowan (population dynamics)
function wilson_cowan_step(E, I_pop; τe=1.0, τi=1.0, wee=10.0, wei=-8.0, wie=10.0, wii=-3.0, Ie=0.0, Ii=0.0, dt=0.1)
    Se(x) = 1.0 / (1.0 + exp(-x))
    dE = (-E + Se(wee*E + wei*I_pop + Ie)) / τe
    dI = (-I_pop + Se(wie*E + wii*I_pop + Ii)) / τi
    return E + dE*dt, I_pop + dI*dt
end

# §7  Prescott (Type I/II switch)
function prescott_step(V, w, I; C=1.0, gL=1.0, EL=-60.0, gw=1.0, Ew=-80.0, τw=50.0, Vth=-40.0, Vr=-60.0, dt=0.1)
    dV = (-gL*(V-EL) - gw*w*(V-Ew) + I) / C
    dw = (1.0/(1+exp(-(V+40)/5)) - w) / τw
    V_new = V + dV*dt; w_new = w + dw*dt
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, w_new, spike
end

# §8  Terman-Wang (oscillatory networks)
function terman_wang_step(x, y, I; τ=1.0, α=3.0, β=4.0, dt=0.1)
    dx = -x + 1.0/(1+exp(-(x-α)/β)) + I
    dy = (-y + x) / τ
    return x + dx*dt, y + dy*dt
end

# §9  Resonate-and-Fire
function resonate_fire_step(x, y, I; ω=0.5, b=0.1, threshold=1.0, dt=0.1)
    dx = b*x - ω*y + I
    dy = ω*x + b*y
    x_new = x + dx*dt; y_new = y + dy*dt
    r = sqrt(x_new^2 + y_new^2)
    spike = r >= threshold
    if spike; x_new = 0.0; y_new = 0.0; end
    return x_new, y_new, spike
end

# §10  SRM0 (Spike Response Model)
function srm0_step(V, t_last_spike, t; τm=20.0, η0=-5.0, τr=5.0, Vth=-50.0, dt=0.1, I=0.0)
    Δt = t - t_last_spike
    η = η0 * exp(-Δt/τr)
    dV = -V/τm + η + I
    V_new = V + dV*dt
    spike = V_new >= Vth
    return V_new, spike
end

# §11  Theta Neuron
function theta_step(θ, I; dt=0.1)
    dθ = 1.0 - cos(θ) + (1.0 + cos(θ)) * I
    return θ + dθ*dt
end

# §12  MAT (Multi-timescale Adaptive Threshold)
function mat_step(V, θ1, θ2, I; τm=10.0, τ1=10.0, τ2=200.0, α1=0.5, α2=0.1, ω=-55.0, dt=0.1)
    dV = -V/τm + I
    dθ1 = -θ1/τ1
    dθ2 = -θ2/τ2
    V_new = V + dV*dt; θ1_new = θ1 + dθ1*dt; θ2_new = θ2 + dθ2*dt
    spike = V_new >= (ω + θ1_new + θ2_new)
    if spike; θ1_new += α1; θ2_new += α2; end
    return V_new, θ1_new, θ2_new, spike
end

# §13  Escape Rate (stochastic)
function escape_rate_step(V, I; τm=20.0, EL=-65.0, ρ0=0.001, ΔV=2.0, Vth=-50.0, Vr=-65.0, dt=0.1)
    dV = (-(V-EL) + I) / τm
    V_new = V + dV*dt
    rate = ρ0 * exp((V_new - Vth) / ΔV) * dt
    spike = rand() < rate
    if spike; V_new = Vr; end
    return V_new, spike
end

# §14  Stochastic IF
function stochastic_if_step(V, I; τm=20.0, σ=0.5, Vth=-50.0, Vr=-65.0, dt=0.1)
    noise = σ * sqrt(dt) * randn()
    dV = (-V/τm + I) * dt + noise
    V_new = V + dV
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, spike
end

# §15  Fractional LIF
function fractional_lif_step(V, history, I; α=0.7, τm=20.0, Vth=-50.0, Vr=-65.0, dt=0.1)
    n = length(history)
    frac_deriv = 0.0
    for k in 1:min(n, 50)
        frac_deriv += (-1)^(k+1) * history[end-k+1] / k^α
    end
    dV = (-V/τm + I + 0.1*frac_deriv) * dt
    V_new = V + dV
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, spike
end

# §16  LTC (Liquid Time-Constant)
function ltc_step(V, I; τ_base=20.0, τ_mod=10.0, EL=-65.0, Vth=-50.0, Vr=-65.0, dt=0.1)
    τ = τ_base + τ_mod * (1.0 / (1.0 + exp(-V/10.0)))
    dV = (-(V-EL) + I) / τ
    V_new = V + dV*dt
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, spike
end

# §17  Wong-Wang (2-population decision model)
function wong_wang_step(S1, S2; I1=0.3, I2=0.3, JN=0.2609, Jext=0.5*1e-3*2400*0.0052, τs=0.1, γ=0.641, dt=0.001)
    x1 = JN*S1 - 0.0497*S2 + I1 + Jext
    x2 = JN*S2 - 0.0497*S1 + I2 + Jext
    H(x) = (0.641*x - 4.0) / (1.0 - exp(-0.4*(0.641*x - 4.0)))
    dS1 = -S1/τs + γ*(1-S1)*H(x1)*dt
    dS2 = -S2/τs + γ*(1-S2)*H(x2)*dt
    return S1 + dS1, S2 + dS2
end

# §18  Gutkin-Ermentrout (Type II)
function gutkin_ermentrout_step(V, w, I; a=0.1, b=0.01, c=-55.0, dt=0.1)
    dV = V*(V-a)*(1-V) - w + I
    dw = b*(V - c*w)
    return V + dV*dt, w + dw*dt
end

# §19  Galves-Löcherbach (point process)
function galves_locherbach_step(V, I; μ=0.1, φ_slope=1.0, Vth=1.0, Vr=0.0, dt=0.1)
    V_new = V * (1-μ*dt) + I*dt
    fire_prob = 1.0 / (1.0 + exp(-φ_slope*(V_new-Vth))) * dt
    spike = rand() < fire_prob
    if spike; V_new = Vr; end
    return V_new, spike
end

# §20  PSN (Parametric Spiking Neuron)
function psn_step(V, I; a=0.02, b=0.2, c=-65.0, d=8.0, dt=0.5)
    dV = 0.04V^2 + 5V + 140 + I
    V_new = V + dV*dt
    spike = V_new >= 30.0
    if spike; V_new = c; end
    return V_new, spike
end

# §21  Sigmoid Rate Model
function sigmoid_rate_step(r, I; τ=10.0, gain=1.0, threshold=0.5, dt=0.1)
    target = 1.0 / (1.0 + exp(-gain*(I-threshold)))
    dr = (target - r) / τ
    return r + dr*dt
end

# §22  SFA (Spike-Frequency Adaptation)
function sfa_step(V, a_adapt, I; τm=20.0, τa=100.0, Δa=0.1, Vth=-50.0, Vr=-65.0, dt=0.1)
    dV = (-V/τm + I - a_adapt)*dt
    da = -a_adapt/τa*dt
    V_new = V + dV; a_new = a_adapt + da
    spike = V_new >= Vth
    if spike; V_new = Vr; a_new += Δa; end
    return V_new, a_new, spike
end

# §23  PLIF (Parametric LIF)
function plif_step(V, I; τ=20.0, α=0.9, β=0.1, Vth=-50.0, Vr=-65.0, dt=0.1)
    V_new = α*V + β*I
    spike = V_new >= Vth
    if spike; V_new = Vr; end
    return V_new, spike
end

# §24  E-prop ALIF
function eprop_alif_step(V, a, e_trace, I; τm=20.0, τa=200.0, β=0.07, Vth=-50.0, Vr=-65.0, dt=0.1)
    dV = (-V/τm + I)*dt
    da = -a/τa*dt
    V_new = V + dV; a_new = a + da
    A = Vth + β*a_new
    spike = V_new >= A
    if spike; V_new = Vr; a_new += 1.0; end
    de = (1.0/τm * max(0.0, 1.0 - abs((V_new-A)/(Vth*0.5))))
    return V_new, a_new, e_trace*0.9 + de*dt, spike
end

# §25  Dendritic NMDA
function dendritic_nmda_step(V, g_nmda, I; τm=20.0, τnmda=100.0, EL=-65.0, Enmda=0.0, Mg=1.0, dt=0.1)
    mg_block = 1.0 / (1.0 + (Mg/3.57)*exp(-0.062*V))
    dV = (-(V-EL) - g_nmda*mg_block*(V-Enmda) + I) / τm
    dg = -g_nmda/τnmda
    return V + dV*dt, g_nmda + dg*dt
end

# §26  Motor Unit
function motor_unit_step(V, force, I; τm=10.0, τf=50.0, Vth=-50.0, Vr=-65.0, f_gain=0.1, dt=0.1)
    dV = (-V/τm + I)*dt
    df = -force/τf*dt
    V_new = V + dV; f_new = force + df
    spike = V_new >= Vth
    if spike; V_new = Vr; f_new += f_gain; end
    return V_new, f_new, spike
end

# §27  Rall Cable
function rall_cable_step(compartments::Vector{Float64}, I_inj::Vector{Float64}; g_c=1.0, τm=20.0, dt=0.1)
    n = length(compartments)
    new_V = copy(compartments)
    for i in 1:n
        coupling = 0.0
        if i > 1; coupling += g_c*(compartments[i-1] - compartments[i]); end
        if i < n; coupling += g_c*(compartments[i+1] - compartments[i]); end
        new_V[i] += (-compartments[i]/τm + coupling + I_inj[i])*dt
    end
    return new_V
end

# §28  Astrocyte (gliotransmission)
function astrocyte_step(Ca, IP3, h; τCa=5.0, τIP3=7.0, τh=15.0, dt=0.1)
    m∞ = IP3/(IP3+0.3)
    n∞ = Ca/(Ca+0.5)
    J_channel = m∞^3 * n∞^3 * h^3 * (1.0 - Ca/0.9)
    J_pump = 0.9*Ca^2/(Ca^2+0.1^2)
    J_leak = 0.05*(1.0-Ca/0.9)
    dCa = (J_channel - J_pump + J_leak) * dt / τCa
    dIP3 = -IP3/τIP3 * dt
    dh = ((0.4*(1-Ca/(Ca+0.3)) - h)/τh)*dt
    return Ca+dCa, IP3+dIP3, h+dh
end

# §29  Tripartite Synapse
function tripartite_step(g_syn, Ca_astro, pre_spike, post_V; τs=5.0, τCa=10.0, gliot_gain=0.5, dt=0.1)
    dg = -g_syn/τs * dt
    if pre_spike; dg += 1.0; end
    dCa = -Ca_astro/τCa*dt + (pre_spike ? 0.1 : 0.0)
    gliot_release = Ca_astro > 0.5 ? gliot_gain : 0.0
    return g_syn+dg+gliot_release*dt, Ca_astro+dCa
end

# §30  Nagumo–Sato refractory map
"""Advance the source equation `y'=k*y-alpha*H(y)+bias+I`, with `H(0)=1`."""
function nagumo_sato_step(y; k=0.6, alpha=1.0, bias=0.2, I=0.0)
    y_new = k*y - alpha*(y >= 0.0 ? 1.0 : 0.0) + bias + I
    return y_new, y_new >= 0.0
end

# §31  SC adaptive-threshold project map
"""Advance the simultaneous two-state SC project recurrence."""
function sc_adaptive_threshold_step(
    x,
    theta;
    k=1.5,
    beta=0.95,
    gamma=0.3,
    theta_spike=0.8,
    x_threshold=0.8,
    I=0.0,
)
    previous_x = x
    activation = 1.0 / (1.0 + exp(-4.0*(x-theta)))
    x_new = clamp(-x + k*activation + I, -5.0, 5.0)
    theta_new = clamp(beta*theta + gamma*(x >= theta_spike ? 1.0 : 0.0), -5.0, 5.0)
    event = x_new >= x_threshold && previous_x < x_threshold
    return x_new, theta_new, event
end

"""Deprecated compatibility spelling for `sc_adaptive_threshold_step`."""
kilinc_bhatt_step(x, theta; kwargs...) = sc_adaptive_threshold_step(x, theta; kwargs...)

# ============================================================
# BATCH SIMULATOR
# ============================================================

function simulate_all_models(duration::Float64=100.0, dt::Float64=0.1)
    n_steps = round(Int, duration/dt)
    results = Dict{String, Vector{Float64}}()

    # LIF variants
    V = -65.0; w = 0.0
    trace = zeros(n_steps)
    for t in 1:n_steps
        V, w, _ = adex_step(V, w, 500.0; dt=dt)
        trace[t] = V
    end
    results["adex"] = trace

    V = -65.0
    trace = zeros(n_steps)
    for t in 1:n_steps
        V, _ = expif_step(V, 15.0; dt=dt)
        trace[t] = V
    end
    results["expif"] = trace

    V = -60.0; ge = 0.0; gi = 0.0
    trace = zeros(n_steps)
    for t in 1:n_steps
        V, ge, gi, _ = coba_lif_step(V, ge, gi, 15.0; dt=dt)
        trace[t] = V
    end
    results["coba_lif"] = trace

    # Decision models
    S1 = 0.1; S2 = 0.1
    trace = zeros(n_steps)
    for t in 1:n_steps
        S1, S2 = wong_wang_step(S1, S2)
        trace[t] = S1
    end
    results["wong_wang"] = trace

    # Population
    E = 0.1; I_pop = 0.1
    trace = zeros(n_steps)
    for t in 1:n_steps
        E, I_pop = wilson_cowan_step(E, I_pop)
        trace[t] = E
    end
    results["wilson_cowan"] = trace

    return results
end

# ============================================================
# BENCHMARK
# ============================================================

function run_benchmarks()
    n_steps = 10000
    dt = 0.1

    models = [
        ("AdEx",          () -> begin V=-65.0; w=0.0; for _ in 1:n_steps; V,w,_=adex_step(V,w,500.0;dt=dt); end; V end),
        ("ExpIF",         () -> begin V=-65.0; for _ in 1:n_steps; V,_=expif_step(V,15.0;dt=dt); end; V end),
        ("COBA-LIF",      () -> begin V=-60.0;ge=0.0;gi=0.0; for _ in 1:n_steps; V,ge,gi,_=coba_lif_step(V,ge,gi,15.0;dt=dt); end; V end),
        ("GLIF",          () -> begin V=-65.0;θ=-50.0; for _ in 1:n_steps; V,θ,_=glif_step(V,θ,15.0;dt=dt); end; V end),
        ("Wilson-Cowan",  () -> begin E=0.1;I=0.1; for _ in 1:n_steps; E,I=wilson_cowan_step(E,I); end; E end),
        ("Prescott",      () -> begin V=-60.0;w=0.0; for _ in 1:n_steps; V,w,_=prescott_step(V,w,15.0;dt=dt); end; V end),
        ("Terman-Wang",   () -> begin x=0.1;y=0.0; for _ in 1:n_steps; x,y=terman_wang_step(x,y,0.5;dt=dt); end; x end),
        ("Resonate&Fire", () -> begin x=0.0;y=0.0; for _ in 1:n_steps; x,y,_=resonate_fire_step(x,y,0.3;dt=dt); end; x end),
        ("Theta",         () -> begin θ=0.0; for _ in 1:n_steps; θ=theta_step(θ,0.5;dt=dt); end; θ end),
        ("MAT",           () -> begin V=-65.0;θ1=0.0;θ2=0.0; for _ in 1:n_steps; V,θ1,θ2,_=mat_step(V,θ1,θ2,15.0;dt=dt); end; V end),
        ("LTC",           () -> begin V=-65.0; for _ in 1:n_steps; V,_=ltc_step(V,15.0;dt=dt); end; V end),
        ("Wong-Wang",     () -> begin S1=0.1;S2=0.1; for _ in 1:n_steps; S1,S2=wong_wang_step(S1,S2); end; S1 end),
        ("Sigmoid Rate",  () -> begin r=0.0; for _ in 1:n_steps; r=sigmoid_rate_step(r,1.0;dt=dt); end; r end),
        ("SFA",           () -> begin V=-65.0;a=0.0; for _ in 1:n_steps; V,a,_=sfa_step(V,a,15.0;dt=dt); end; V end),
        ("PLIF",          () -> begin V=-65.0; for _ in 1:n_steps; V,_=plif_step(V,15.0;dt=dt); end; V end),
        ("Rall Cable 8",  () -> begin c=fill(-65.0,8); I=zeros(8); I[1]=15.0; for _ in 1:n_steps; c=rall_cable_step(c,I;dt=dt); end; c[1] end),
        ("Astrocyte",     () -> begin Ca=0.1;IP3=0.5;h=0.5; for _ in 1:n_steps; Ca,IP3,h=astrocyte_step(Ca,IP3,h;dt=dt); end; Ca end),
        ("Nagumo–Sato",   () -> begin y=0.1; for _ in 1:n_steps; y,_=nagumo_sato_step(y); end; y end),
        ("SC adaptive",   () -> begin x=0.0;theta=0.0; for _ in 1:n_steps; x,theta,_=sc_adaptive_threshold_step(x,theta); end; x end),
    ]

    println("=" ^ 55)
    println("SC-NeuroCore Julia Neuron Zoo — 31 Models Benchmark")
    println("=" ^ 55)

    for (name, fn) in models
        fn()  # warmup
        t0 = time()
        for _ in 1:100
            fn()
        end
        elapsed = (time()-t0) * 10.0  # ms per solve
        @printf("%-20s  10k steps × 100:  %8.3f ms\n", name, elapsed)
    end

    println("=" ^ 55)
    println("30 neuron models, 30 step functions")
end

end # module NeuronZoo
