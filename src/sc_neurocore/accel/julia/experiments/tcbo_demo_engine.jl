# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/tcbo_demo_engine

module TcboDemoEngineAccel

using Statistics, LinearAlgebra

mutable struct TCBODemoEngineState
    name::Float64
    description::Float64
    duration_s::Float64
    K_scale::Float64
    noise_amplitude::Float64
    use_controller::Float64
    phase_scramble::Float64
    alpha_boost::Float64
    coupling_decay_rate::Float64
    N::Float64
    dt::Float64
    _rng::Float64
    _seed::Float64
    omega::Float64
    _K_base::Float64
end

function TCBODemoEngineState()
    TCBODemoEngineState(0.0, 0.0, 10.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function set_coupling_scale(s::TCBODemoEngineState, scale)
    s.K = s._K_base * scale
end

function apply_anesthesia(s::TCBODemoEngineState, strength)
    s.K *= 1.0 - strength
    s.theta = s._rng.uniform(0, 2 * pi, s.N)
    s.noise_amplitude *= 10.0
end

function apply_alpha_boost(s::TCBODemoEngineState, factor)
    if s.N >= 3
        s.K[1, :] *= factor
        s.K[:, 1] *= factor
        np.fill_diagonal(s.K, 0)
end

function apply_coupling_decay(s::TCBODemoEngineState, rate)
    s.K *= 1.0 - rate
end

function step(s::TCBODemoEngineState, perturbation, Any]])
    dtheta = s.omega.copy()
    # Kuramoto coupling: Σ K_nm sin(θ_m - θ_n)
    for n in 1:s.N
        coupling = 0.0
        for m in 1:s.N
            if m != n
                coupling += s.K[n, m] * sin(s.theta[m] - s.theta[n])
        dtheta[n] += coupling
    # Noise
    dtheta += s.noise_amplitude * s._rng.randn(s.N)
    # External perturbation
    if perturbation is ! nothing
        dtheta += perturbation
    s.theta = (s.theta + dtheta * s.dt) % (2 * pi)
    s._step_count += 1
    return s.theta.copy()
end

function run(s::TCBODemoEngineState, n_steps)
    history = zeros((n_steps, s.N))
    for i in 1:n_steps
        history[i] = s.step()
    return history
end

function get_order_parameter(s::TCBODemoEngineState)
    return _compute_order_parameter(s.theta)
end

function reset(s::TCBODemoEngineState, seed)
    if seed is ! nothing
        s._rng = np.random.RandomState(seed)
    s.theta = s._rng.uniform(0, 2 * pi, s.N)
    s.K = s._K_base.copy()
    s.noise_amplitude = 0.3
    s._step_count = 0
end

function step(s::TCBODemoEngineState, p_h1, kappa, dt)
    error = max(0.0, s.tau_h1 - p_h1)
    s._integral += error * dt
    # Anti-windup
    s._integral = clamp(s._integral, 0, 10.0)
    delta = s.Kp * error + s.Ki * s._integral
    new_kappa = kappa + delta * dt
    return float(clamp(new_kappa, s.kappa_min, s.kappa_max))
end

function reset(s::TCBODemoEngineState)
    s._integral = 0.0
end

function to_dict(s::TCBODemoEngineState)
    return {
        "step": s.step,
        "time_s": round(s.time_s, 4),
        "phases": [round(p, 4) for p in s.phases],
        "R_global": round(s.R_global, 4),
        "p_h1": round(s.p_h1, 4),
        "gate_open": s.gate_open,
        "is_conscious": s.is_conscious,
        "kappa": round(s.kappa, 4),
        "has_tcbo": s.has_tcbo,
    }
end

function get_scenarios(s::TCBODemoEngineState)
    return {
        name.value: {
            "name": cfg.name,
            "description": cfg.description,
            "duration_s": cfg.duration_s,
        }
        for name, cfg in SCENARIOS.items()
    }
end

function start_scenario(s::TCBODemoEngineState, name)
    try
        scenario_name = ScenarioName(name)
    except ValueError
        raise ValueError(
            f"Unknown scenario: {name}. Available: {[s.value for s in ScenarioName]}"
        )
    cfg = SCENARIOS[scenario_name]
    s._current_scenario = name
    s._scenario_cfg = cfg
    # Reset generator
    s.gen.reset(seed=s._seed)
    s.gen.set_coupling_scale(cfg.K_scale)
    s.gen.noise_amplitude = cfg.noise_amplitude
    if cfg.phase_scramble
        s.gen.theta = np.random.RandomState(s._seed + 1).uniform(0, 2 * pi, s.N)
    if cfg.alpha_boost > 0
        s.gen.apply_alpha_boost(cfg.alpha_boost)
    s.controller.reset()
    s.kappa = cfg.K_scale
    s.p_h1 = 0.0
    s._step_count = 0
    s._max_steps = int(cfg.duration_s / s.dt)
    s._phase_history.clear()
    s._snapshots.clear()
    s.is_running = true
    return {"scenario": name, "max_steps": s._max_steps, "dt": s.dt}
end

function step(s::TCBODemoEngineState)
    if ! s.is_running
        raise RuntimeError("No scenario running")
    cfg = s._scenario_cfg
    # Apply coupling decay if configured
    if cfg && cfg.coupling_decay_rate > 0
        s.gen.apply_coupling_decay(cfg.coupling_decay_rate)
    # Kuramoto step
    phases = s.gen.step()
    s._phase_history = push!(, phases)
    # Keep bounded history
    if length(s._phase_history) > 200
        s._phase_history = s._phase_history[-100:]
    # Compute observables
    R = s.gen.get_order_parameter()
    history_arr = collect(s._phase_history)
    s.p_h1 = _compute_p_h1_lightweight(history_arr, s.TAU_H1)
    gate_open = s.p_h1 > s.TAU_H1
    # PI controller
    if cfg && cfg.use_controller
        new_kappa = s.controller.step(s.p_h1, s.kappa, s.dt)
        if new_kappa > s.kappa
            s.gen.set_coupling_scale(new_kappa)
        s.kappa = new_kappa
    s._step_count += 1
    if s._step_count >= s._max_steps
        s.is_running = false
    snap = TCBODemoSnapshot(
        step=s._step_count,
        time_s=s._step_count * s.dt,
        phases=phases.tolist(),
        R_global=R,
        p_h1=s.p_h1,
        gate_open=gate_open,
        is_conscious=gate_open,
        kappa=s.kappa,
        has_tcbo=false,
    )
    s._snapshots = push!(, snap)
    return snap
end

function run_scenario(s::TCBODemoEngineState)
    self,
    name: str,
    duration_s: Optional[float] = nothing,
    subsample: int = 100,
    ) -> List[TCBODemoSnapshot]
    s.start_scenario(name)
    if duration_s is ! nothing
        s._max_steps = int(duration_s / s.dt)
    results = []
    for i in 1:s._max_steps
        snap = s.step()
        if i % subsample == 0
            results = push!(, snap)
    return results
end

function get_state(s::TCBODemoEngineState)
    return {
        "running": s.is_running,
        "scenario": s._current_scenario,
        "step": s._step_count,
        "p_h1": round(s.p_h1, 4),
        "kappa": round(s.kappa, 4),
        "R_global": round(s.gen.get_order_parameter(), 4),
        "gate_open": s.p_h1 > s.TAU_H1,
    }
end

function get_history(s::TCBODemoEngineState, last_n)
    return [s.to_dict() for s in s._snapshots[-last_n:]]
end

function reset(s::TCBODemoEngineState)
    s.gen.reset(seed=s._seed)
    s.controller.reset()
    s.p_h1 = 0.0
    s.kappa = 1.0
    s.is_running = false
    s._current_scenario = nothing
    s._step_count = 0
    s._phase_history.clear()
    s._snapshots.clear()
end

function get_tcbo_demo_engine()
    global _engine
    if _engine is nothing
        _engine = TCBODemoEngine()
    return _engine
end

function reset_tcbo_demo_engine()
    global _engine
    _engine = nothing
end

end # module TcboDemoEngineAccel
