# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete Julia ODE / Scientific Computing Suite

using DifferentialEquations, LinearAlgebra, Statistics, Printf

# ============================================================
# §1  LIF (Leaky Integrate-and-Fire) ODE
# ============================================================
function lif_ode!(du, u, p, t)
    τ, V_rest, V_thresh, I_ext = p
    du[1] = (-(u[1] - V_rest) + I_ext) / τ
end

function solve_lif(; τ=20.0, V_rest=-65.0, V_thresh=-50.0, I_ext=15.0, tspan=(0.0, 100.0))
    u0 = [V_rest]
    p = (τ, V_rest, V_thresh, I_ext)
    prob = ODEProblem(lif_ode!, u0, tspan, p)
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §2  Izhikevich Neuron ODE (4 modes)
# ============================================================
function izhikevich_ode!(du, u, p, t)
    a, b, I = p
    v, w = u
    du[1] = 0.04v^2 + 5v + 140 - w + I
    du[2] = a * (b * v - w)
end

function solve_izhikevich(; mode=:RS, I=10.0, tspan=(0.0, 200.0))
    params = Dict(
        :RS => (a=0.02, b=0.2, c=-65.0, d=8.0),
        :IB => (a=0.02, b=0.2, c=-55.0, d=4.0),
        :CH => (a=0.02, b=0.2, c=-50.0, d=2.0),
        :FS => (a=0.1,  b=0.2, c=-65.0, d=2.0),
    )
    p = params[mode]
    u0 = [p.c, p.b * p.c]
    prob = ODEProblem(izhikevich_ode!, u0, tspan, (p.a, p.b, I))

    condition(u, t, integrator) = u[1] - 30.0
    affect!(integrator) = begin
        integrator.u[1] = p.c
        integrator.u[2] += p.d
    end
    cb = ContinuousCallback(condition, affect!)
    solve(prob, Tsit5(), callback=cb, saveat=0.1)
end

# ============================================================
# §3  Hodgkin-Huxley (Full Biophysical)
# ============================================================
function hodgkin_huxley!(du, u, p, t)
    V, m, h, n = u
    gNa, gK, gL, ENa, EK, EL, Cm, I_ext = p

    αm = 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
    βm = 4.0 * exp(-(V + 65.0) / 18.0)
    αh = 0.07 * exp(-(V + 65.0) / 20.0)
    βh = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
    αn = 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
    βn = 0.125 * exp(-(V + 65.0) / 80.0)

    INa = gNa * m^3 * h * (V - ENa)
    IK  = gK  * n^4 * (V - EK)
    IL  = gL  * (V - EL)

    du[1] = (I_ext - INa - IK - IL) / Cm
    du[2] = αm * (1 - m) - βm * m
    du[3] = αh * (1 - h) - βh * h
    du[4] = αn * (1 - n) - βn * n
end

function solve_hh(; I_ext=10.0, tspan=(0.0, 50.0))
    p = (120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0, I_ext)
    u0 = [-65.0, 0.05, 0.6, 0.32]
    prob = ODEProblem(hodgkin_huxley!, u0, tspan, p)
    solve(prob, Tsit5(), saveat=0.01)
end

# ============================================================
# §4  SC Equilibrium Dynamics
# ============================================================
function sc_equilibrium!(du, u, p, t)
    τ_eq, target = p
    du[1] = (target - u[1]) / τ_eq + 0.01 * sin(2π * t / 10)
end

function solve_sc_equilibrium(; target=0.5, τ=5.0, tspan=(0.0, 50.0))
    prob = ODEProblem(sc_equilibrium!, [0.0], tspan, (τ, target))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §5  FitzHugh-Nagumo (Reduced 2D oscillator)
# ============================================================
function fitzhugh_nagumo!(du, u, p, t)
    a, b, τ, I = p
    v, w = u
    du[1] = v - v^3 / 3 - w + I
    du[2] = (v + a - b * w) / τ
end

function solve_fhn(; a=0.7, b=0.8, τ=12.5, I=0.5, tspan=(0.0, 200.0))
    prob = ODEProblem(fitzhugh_nagumo!, [-1.0, -0.5], tspan, (a, b, τ, I))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §6  AdEx (Adaptive Exponential Integrate-and-Fire)
# ============================================================
function adex!(du, u, p, t)
    C, gL, EL, ΔT, VT, τw, a, I = p
    V, w = u
    du[1] = (-gL * (V - EL) + gL * ΔT * exp((V - VT) / ΔT) - w + I) / C
    du[2] = (a * (V - EL) - w) / τw
end

function solve_adex(; I=500.0, tspan=(0.0, 200.0))
    p = (281.0, 30.0, -70.6, 2.0, -50.4, 144.0, 4.0, I)
    u0 = [-70.6, 0.0]
    prob = ODEProblem(adex!, u0, tspan, p)
    condition(u, t, int) = u[1] - 20.0
    affect!(int) = begin; int.u[1] = -70.6; int.u[2] += 80.5; end
    cb = ContinuousCallback(condition, affect!)
    solve(prob, Tsit5(), callback=cb, saveat=0.1)
end

# ============================================================
# §7  Morris-Lecar (2D Biophysical)
# ============================================================
function morris_lecar!(du, u, p, t)
    C, gCa, gK, gL, VCa, VK, VL, V1, V2, V3, V4, φ, I = p
    V, w = u
    m∞ = 0.5 * (1 + tanh((V - V1) / V2))
    w∞ = 0.5 * (1 + tanh((V - V3) / V4))
    τw = 1.0 / (φ * cosh((V - V3) / (2V4)))
    du[1] = (-gCa * m∞ * (V - VCa) - gK * w * (V - VK) - gL * (V - VL) + I) / C
    du[2] = (w∞ - w) / τw
end

function solve_morris_lecar(; I=90.0, tspan=(0.0, 200.0))
    p = (20.0, 4.0, 8.0, 2.0, 120.0, -84.0, -60.0, -1.2, 18.0, 12.0, 17.4, 0.0667, I)
    u0 = [-60.0, 0.01]
    prob = ODEProblem(morris_lecar!, u0, tspan, p)
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §8  Coupled Oscillator Network (Gamma Oscillations)
# ============================================================
function coupled_oscillators!(du, u, p, t)
    N, ω, K = p
    for i in 1:N
        coupling = 0.0
        for j in 1:N
            coupling += sin(u[j] - u[i])
        end
        du[i] = ω[i] + K / N * coupling
    end
end

function solve_gamma_oscillation(; N=8, K=2.0, tspan=(0.0, 100.0))
    ω = 2π * (40.0 .+ 5.0 * randn(N))  # ~40 Hz gamma
    u0 = 2π * rand(N)
    prob = ODEProblem(coupled_oscillators!, u0, tspan, (N, ω, K))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §9  Memristor I-V Curve (Device Physics)
# ============================================================
function memristor_dynamics!(du, u, p, t)
    Ron, Roff, μ, D, V_amplitude, freq = p
    x = u[1]  # state variable [0, 1]
    V = V_amplitude * sin(2π * freq * t)
    R = Ron * x + Roff * (1 - x)
    I = V / R
    du[1] = μ * Ron / D^2 * I * (1 - (2x - 1)^2)
end

function solve_memristor(; V_amp=1.0, freq=1.0, tspan=(0.0, 5.0))
    p = (100.0, 16000.0, 1e-14, 10e-9, V_amp, freq)
    u0 = [0.5]
    prob = ODEProblem(memristor_dynamics!, u0, tspan, p)
    solve(prob, Rodas5(), saveat=0.001)
end

# ============================================================
# §10  Phase Field Dynamics (PDE-like via MOL)
# ============================================================
function phase_field!(du, u, p, t)
    N, ε, γ = p
    dx = 1.0 / N
    for i in 1:N
        left = i > 1 ? u[i-1] : u[N]
        right = i < N ? u[i+1] : u[1]
        laplacian = (left - 2u[i] + right) / dx^2
        du[i] = ε^2 * laplacian + u[i] - u[i]^3 + γ
    end
end

function solve_phase_field(; N=64, ε=0.1, γ=0.0, tspan=(0.0, 10.0))
    u0 = 0.1 * randn(N)
    prob = ODEProblem(phase_field!, u0, tspan, (N, ε, γ))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §11  Predictive Coding (Free Energy Minimization)
# ============================================================
function predictive_coding!(du, u, p, t)
    N, σ, μ_prior = p
    half = N ÷ 2
    for i in 1:half
        prediction_error = u[i] - u[half + i]
        du[i] = -prediction_error / σ^2
        du[half + i] = prediction_error / σ^2 + (μ_prior - u[half + i]) * 0.1
    end
end

function solve_predictive_coding(; N=16, σ=1.0, tspan=(0.0, 50.0))
    u0 = randn(N)
    prob = ODEProblem(predictive_coding!, u0, tspan, (N, σ, 0.0))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §12  Homeostatic Regulation (PI Controller)
# ============================================================
function homeostatic_pi!(du, u, p, t)
    target_rate, Kp, Ki = p
    rate, integral = u
    error = target_rate - rate
    du[1] = -0.1 * rate + Kp * error + Ki * integral
    du[2] = error
end

function solve_homeostasis(; target=5.0, Kp=0.5, Ki=0.1, tspan=(0.0, 100.0))
    prob = ODEProblem(homeostatic_pi!, [0.0, 0.0], tspan, (target, Kp, Ki))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §13  Quantum Noise Model (Lindblad-like)
# ============================================================
function quantum_decoherence!(du, u, p, t)
    T1, T2, ω = p
    ρ_00, ρ_01_re, ρ_01_im, ρ_11 = u
    du[1] = ρ_11 / T1
    du[2] = -ρ_01_re / T2 - ω * ρ_01_im
    du[3] = -ρ_01_im / T2 + ω * ρ_01_re
    du[4] = -ρ_11 / T1
end

function solve_decoherence(; T1=50.0, T2=20.0, ω=1.0, tspan=(0.0, 100.0))
    u0 = [0.0, 0.5, 0.0, 1.0]  # initially excited
    prob = ODEProblem(quantum_decoherence!, u0, tspan, (T1, T2, ω))
    solve(prob, Tsit5(), saveat=0.1)
end

# ============================================================
# §14  STDP Timing Curve (Continuous Exponential Kernel)
# ============================================================
function stdp_kernel(Δt; A_plus=0.01, A_minus=0.012, τ_plus=20.0, τ_minus=20.0)
    if Δt > 0
        return A_plus * exp(-Δt / τ_plus)
    else
        return -A_minus * exp(Δt / τ_minus)
    end
end

function compute_stdp_curve(; dt_range=-50.0:0.5:50.0)
    return [stdp_kernel(dt) for dt in dt_range]
end

# ============================================================
# §15  Diffusion Process (Brownian Motion SDE)
# ============================================================
function diffusion_drift!(du, u, p, t)
    μ, = p
    du[1] = μ * (0.5 - u[1])
end

function diffusion_noise!(du, u, p, t)
    _, σ = p
    du[1] = σ
end

function solve_diffusion(; μ=0.5, σ=0.1, tspan=(0.0, 100.0))
    prob = SDEProblem(diffusion_drift!, diffusion_noise!, [0.5], tspan, (μ, σ))
    solve(prob, EM(), dt=0.01, saveat=0.1)
end

# ============================================================
# BENCHMARK
# ============================================================
function run_benchmarks()
    println("=" ^ 55)
    println("SC-NeuroCore Julia ODE Suite — Full Benchmark")
    println("=" ^ 55)

    solvers = [
        ("§1  LIF ODE",          () -> solve_lif()),
        ("§2  Izhikevich RS",    () -> solve_izhikevich(mode=:RS)),
        ("§3  Hodgkin-Huxley",   () -> solve_hh()),
        ("§4  SC Equilibrium",   () -> solve_sc_equilibrium()),
        ("§5  FitzHugh-Nagumo",  () -> solve_fhn()),
        ("§6  AdEx IF",          () -> solve_adex()),
        ("§7  Morris-Lecar",     () -> solve_morris_lecar()),
        ("§8  Gamma Oscillation",() -> solve_gamma_oscillation()),
        ("§9  Memristor I-V",    () -> solve_memristor()),
        ("§10 Phase Field 64",   () -> solve_phase_field()),
        ("§11 Predictive Coding",() -> solve_predictive_coding()),
        ("§12 Homeostasis PI",   () -> solve_homeostasis()),
        ("§13 Quantum Decohere", () -> solve_decoherence()),
        ("§14 STDP Curve",       () -> compute_stdp_curve()),
    ]

    for (name, solver) in solvers
        solver()  # warmup / compile
        t0 = time()
        for _ in 1:100
            solver()
        end
        t1 = time()
        @printf("%-25s  %8.3f ms/solve\n", name, (t1 - t0) * 10.0)
    end

    println("=" ^ 55)
    println("14 ODE models, 15 solver functions")
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Printf
    run_benchmarks()
end
