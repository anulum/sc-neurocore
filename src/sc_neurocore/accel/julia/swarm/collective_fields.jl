# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for swarm/collective_fields

module CollectiveFieldsAccel

using Statistics, LinearAlgebra

mutable struct CollectiveFieldsState
    grid_size::Float64
    diffusion_rate::Float64
    decay_rate::Float64
    emotional_coupling::Float64
    symbolic_decay::Float64
    seed::Float64
    cfg::Float64
    env_width::Float64
    env_height::Float64
    n_agents::Float64
    rng::Float64
    chemical_field::Float64
    emotional_field::Float64
    symbolic_field::Float64
end

function CollectiveFieldsState()
    CollectiveFieldsState(50.0, 0.1, 0.05, 0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function _to_grid(s::CollectiveFieldsState, x, y)
    gs = s.cfg.grid_size
    col = int(clamp(x / s.env_width * gs, 0, gs - 1))
    row = int(clamp(y / s.env_height * gs, 0, gs - 1))
    return row, col
end

function diffuse(s::CollectiveFieldsState, dt)
    lap = _apply_laplacian(s.chemical_field)
    s.chemical_field += s.cfg.diffusion_rate * dt * lap
    s.chemical_field *= 1.0 - s.cfg.decay_rate * dt
    clamp(s.chemical_field, 0, nothing, out=s.chemical_field)
end

function deposit_chemical(s::CollectiveFieldsState, x, y, amount)
    if amount <= 0
        return
    r, c = s._to_grid(x, y)
    s.chemical_field[r, c] += amount
end

function get_chemical_gradient(s::CollectiveFieldsState, x, y)
    r, c = s._to_grid(x, y)
    gs = s.cfg.grid_size
    f = s.chemical_field
    # Central differences with boundary clamp
    dc = (f[r, min(c + 1, gs - 1)] - f[r, max(c - 1, 0)]) * 0.5
    dr = (f[min(r + 1, gs - 1), c] - f[max(r - 1, 0), c]) * 0.5
    # Map grid gradient -> world gradient direction
    dx = float(dc)
    dy = float(dr)
    norm = sqrt(dx * dx + dy * dy) + 1e-12
    return dx / norm, dy / norm
end

function synchronize_emotions(s::CollectiveFieldsState, coupling)
    if coupling is nothing
        coupling = s.cfg.emotional_coupling
    mean_emotion = s.emotional_field.mean(axis=0)
    s.emotional_field += coupling * (mean_emotion - s.emotional_field)
end

function get_symbolic_at(s::CollectiveFieldsState, x, y)
    r, c = s._to_grid(x, y)
    return s.symbolic_field[r, c].copy()
end

function deposit_symbolic(s::CollectiveFieldsState, x, y, channel, amount)
    r, c = s._to_grid(x, y)
    s.symbolic_field[r, c, channel] += amount
end

function update(s::CollectiveFieldsState, agents, env, dt)
    # Push agent emotions into the field
    for idx, agent in enumerate(agents)
        if idx < s.n_agents
            s.emotional_field[idx] = agent.emotions
    s.diffuse(dt)
    s.synchronize_emotions()
    # Symbolic decay
    s.symbolic_field *= 1.0 - s.cfg.symbolic_decay * dt
    # Pull updated emotions back to agents
    for idx, agent in enumerate(agents)
        if idx < s.n_agents
            agent.emotions = s.emotional_field[idx].copy()
end

end # module CollectiveFieldsAccel
