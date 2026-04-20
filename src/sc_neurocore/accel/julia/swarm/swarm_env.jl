# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for swarm/swarm_env

module SwarmEnvAccel

using Statistics, LinearAlgebra

mutable struct SwarmEnvironmentState
    width::Float64
    height::Float64
    n_agents::Float64
    n_obstacles::Float64
    n_targets::Float64
    boundary_mode::Float64
    capture_radius::Float64
    respawn_targets::Float64
    agent_config::Float64
    seed::Float64
    cfg::Float64
    rng::Float64
    obstacles::Float64
    targets::Float64
    targets_captured::Float64
end

function SwarmEnvironmentState()
    SwarmEnvironmentState(100.0, 100.0, 20.0, 5.0, 3.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function _random_target_pos(s::SwarmEnvironmentState)
    return s.rng.uniform([5, 5], [s.cfg.width - 5, s.cfg.height - 5])
end

function _apply_boundary(s::SwarmEnvironmentState, agent)
    if s.cfg.boundary_mode == "wrap"
        agent.position[0] %= s.cfg.width
        agent.position[1] %= s.cfg.height
    else:  # clamp
        agent.position[0] = clamp(agent.position[0], 0, s.cfg.width)
        agent.position[1] = clamp(agent.position[1], 0, s.cfg.height)
end

function get_positions(s::SwarmEnvironmentState)
    return collect([a.position for a in s.agents])
end

function get_headings(s::SwarmEnvironmentState)
    return collect([a.heading for a in s.agents])
end

function get_pairwise_distances(s::SwarmEnvironmentState)
    pos = s.get_positions()
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    return sqrt((diff^2).sum(axis=-1))
end

function get_neighbor_distances(s::SwarmEnvironmentState, agent_idx, k)
    pos = s.get_positions()
    diff = pos - pos[agent_idx]
    dists = sqrt((diff^2).sum(axis=-1))
    dists[agent_idx] = Inf  # exclude self
    sorted_d = sort(dists)
    out = zeros(k)
    n = min(k, length(sorted_d) - 1)
    out[:n] = sorted_d[:n]
    return out
end

function get_obstacle_distances(s::SwarmEnvironmentState, agent_idx, k)
    pos = s.agents[agent_idx].position
    centers = s.obstacles[:, :2]
    radii = s.obstacles[:, 2]
    dists = sqrt(((centers - pos) ^ 2).sum(axis=-1)) - radii
    sorted_d = sort(dists)
    out = zeros(k)
    n = min(k, length(sorted_d))
    out[:n] = sorted_d[:n]
    return out
end

function get_target_distances(s::SwarmEnvironmentState, agent_idx, k)
    pos = s.agents[agent_idx].position
    dists = sqrt(((s.targets - pos) ^ 2).sum(axis=-1))
    sorted_d = sort(dists)
    out = zeros(k)
    n = min(k, length(sorted_d))
    out[:n] = sorted_d[:n]
    return out
end

function step(s::SwarmEnvironmentState, dt, fields)
    cfg = s.cfg
    for idx, agent in enumerate(s.agents)
        # Build 20-channel sensory vector
        sensory = zeros(agent.cfg.n_sensory)
        nbr_dist = s.get_neighbor_distances(idx, k=8)
        sensory[0:8] = clamp(nbr_dist / max(cfg.width, cfg.height), 0, 1)
        od = s.get_obstacle_distances(idx, k=3)
        sensory[8:11] = clamp(od / 50.0, -1, 1)
        td = s.get_target_distances(idx, k=2)
        sensory[11:13] = clamp(td / max(cfg.width, cfg.height), 0, 1)
        if fields is ! nothing
            gx, gy = fields.get_chemical_gradient(agent.position[0], agent.position[1])
            sensory[13:15] = [gx, gy]
            sym = fields.get_symbolic_at(agent.position[0], agent.position[1])
            sensory[15:17] = sym
            sensory[17:19] = agent.emotions[:2]
            sensory[19] = agent.chemical_output
        # else: zeros (safe defaults)
        speed, turn = agent.think(sensory)
        agent.act(speed * dt, turn * dt)
        s._apply_boundary(agent)
        # Chemical deposit
        if fields is ! nothing
            fields.deposit_chemical(
                agent.position[0], agent.position[1], agent.chemical_output * dt
            )
    # --- Target capture ---
    positions = s.get_positions()
    for t_idx in 1:length(s.targets)
        dists = sqrt(((positions - s.targets[t_idx]) ^ 2).sum(axis=-1))
        if dists.min() < cfg.capture_radius
            s.targets_captured += 1
            if cfg.respawn_targets
                s.targets[t_idx] = s._random_target_pos()
    # --- Update fields ---
    if fields is ! nothing
        fields.update(s.agents, self, dt)
    s.step_count += 1
end

function get_state(s::SwarmEnvironmentState)
    return {
        "step": s.step_count,
        "positions": s.get_positions().tolist(),
        "headings": s.get_headings().tolist(),
        "obstacles": s.obstacles.tolist(),
        "targets": s.targets.tolist(),
        "targets_captured": s.targets_captured,
    }
end

end # module SwarmEnvAccel
