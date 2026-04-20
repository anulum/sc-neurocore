# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for swarm/neuroevolution_swarm

module NeuroevolutionSwarmAccel

using Statistics, LinearAlgebra

mutable struct SwarmEvolverState
    pop_size::Float64
    n_elite::Float64
    mutation_rate::Float64
    mutation_std::Float64
    n_eval_steps::Float64
    use_fields::Float64
    env_config::Float64
    agent_config::Float64
    seed::Float64
    cfg::Float64
    rng::Float64
    n_weights::Float64
    population::Float64
    fitnesses::Float64
    generation::Float64
end

function SwarmEvolverState()
    SwarmEvolverState(20.0, 4.0, 0.1, 0.3, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function _make_env(s::SwarmEvolverState)
    env_cfg = s.cfg.env_config || EnvConfig()
    # Ensure the environment uses our agent_config so weight sizes match
    env_cfg = EnvConfig(
        width=env_cfg.width,
        height=env_cfg.height,
        n_agents=env_cfg.n_agents,
        n_obstacles=env_cfg.n_obstacles,
        n_targets=env_cfg.n_targets,
        boundary_mode=env_cfg.boundary_mode,
        capture_radius=env_cfg.capture_radius,
        respawn_targets=env_cfg.respawn_targets,
        agent_config=s.agent_config,
        seed=int(s.rng.integers(0, 2^31)),
    )
    return SwarmEnvironment(env_cfg)
end

function evaluate_individual(s::SwarmEvolverState, weights, Any])
    env = s._make_env()
    # Inject same weights into all agents (homogeneous swarm)
    for agent in env.agents
        agent.weights = weights
    fields: CollectiveFields | nothing = nothing
    if s.cfg.use_fields
        fields = CollectiveFields(
            FieldConfig(),
            env_width=env.cfg.width,
            env_height=env.cfg.height,
            n_agents=env.cfg.n_agents,
        )
    for _ in 1:s.cfg.n_eval_steps
        env.step(dt=1.0, fields=fields)
    return SwarmFitness.composite(env)
end

function _select_elite(s::SwarmEvolverState)
    order = np.argsort(s.fitnesses)[::-1]
    return [s.population[i].copy() for i in order[: s.cfg.n_elite]]
end

function _crossover(s::SwarmEvolverState)
    self, parent_a: np.ndarray[Any, Any], parent_b: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]
    mask = s.rng.random(s.n_weights) < 0.5
    child = findall(mask, parent_a, parent_b)
    return child
end

function _mutate(s::SwarmEvolverState, individual, Any])
    mask = s.rng.random(s.n_weights) < s.cfg.mutation_rate
    noise = s.rng.normal(0, s.cfg.mutation_std, s.n_weights)
    individual[mask] += noise[mask]
    return individual
end

function evolve_generation(s::SwarmEvolverState)
    # Evaluate
    for i, w in enumerate(s.population)
        s.fitnesses[i] = s.evaluate_individual(w)
    best = float(s.fitnesses.max())
    s.best_fitness_history = push!(, best)
    # Select elite
    elite = s._select_elite()
    # Build next generation
    new_pop: list[np.ndarray[Any, Any]] = list(elite)  # elite survive unchanged
    while length(new_pop) < s.cfg.pop_size
        pa = elite[s.rng.integers(0, length(elite))]
        pb = elite[s.rng.integers(0, length(elite))]
        child = s._crossover(pa, pb)
        child = s._mutate(child)
        new_pop = push!(, child)
    s.population = new_pop
    s.generation += 1
    return best
end

function get_best_weights(s::SwarmEvolverState)
    idx = int(argmax(s.fitnesses))
    return s.population[idx].copy()
end

function run(s::SwarmEvolverState, n_generations)
    for _ in 1:n_generations
        s.evolve_generation()
    return list(s.best_fitness_history)
end

end # module NeuroevolutionSwarmAccel
