# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for swarm/agent

module AgentAccel

using Statistics, LinearAlgebra

mutable struct SwarmAgentState
    n_sensory::Float64
    n_hidden::Float64
    n_motor::Float64
    membrane_decay::Float64
    threshold::Float64
    max_speed::Float64
    seed::Float64
    cfg::Float64
    agent_id::Float64
    W_in::Float64
    W_rec::Float64
    W_out::Float64
    membrane::Float64
    firing_rate::Float64
    position::Float64
end

function SwarmAgentState()
    SwarmAgentState(20.0, 16.0, 2.0, 0.9, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function n_weights(s::SwarmAgentState)
    c = s.cfg
    return c.n_hidden * c.n_sensory + c.n_hidden * c.n_hidden + c.n_motor * c.n_hidden
end

function weights(s::SwarmAgentState)
    return vcat(
        [
            s.W_in.ravel(),
            s.W_rec.ravel(),
            s.W_out.ravel(),
        ]
    )
end

function weights(s::SwarmAgentState, flat, Any])
    c = s.cfg
    if flat.size != s.n_weights
        raise ValueError(f"Expected {s.n_weights} weights, got {flat.size}")
    offset = 0
    size_in = c.n_hidden * c.n_sensory
    s.W_in = flat[offset : offset + size_in].reshape(c.n_hidden, c.n_sensory).copy()
    offset += size_in
    size_rec = c.n_hidden * c.n_hidden
    s.W_rec = flat[offset : offset + size_rec].reshape(c.n_hidden, c.n_hidden).copy()
    offset += size_rec
    size_out = c.n_motor * c.n_hidden
    s.W_out = flat[offset : offset + size_out].reshape(c.n_motor, c.n_hidden).copy()
end

function think(s::SwarmAgentState, sensory, Any])
    c = s.cfg
    inp = np.asarray(sensory, dtype=np.float64).ravel()[: c.n_sensory]
    # Membrane integration
    s.membrane = (
        c.membrane_decay * s.membrane + s.W_in @ inp + s.W_rec @ s.firing_rate  # type: ignore[assignment]
    )
    # Soft spike (sigmoid pseudo-rate)
    spike_prob = 1.0 / (1.0 + exp(-(s.membrane - c.threshold)))
    s.firing_rate = 0.8 * s.firing_rate + 0.2 * spike_prob  # type: ignore[assignment]
    # Reset membrane where spike probability high
    s.membrane *= 1.0 - spike_prob
    # Motor readout
    motor = s.W_out @ s.firing_rate
    speed = (tanh(motor[0]) + 1.0) * 0.5 * c.max_speed  # [0, max_speed]
    turn = tanh(motor[1]) * pi  # [-pi, pi]
    # Side-effect: chemical output from last sensory channel
    s.chemical_output = float(clamp(sensory[-1] if length(sensory) > 19 else 0.0, 0, 1))
    return float(speed), float(turn)
end

function act(s::SwarmAgentState, speed, turn)
    s.heading = (s.heading + turn) % (2 * pi)
    dx = speed * cos(s.heading)
    dy = speed * sin(s.heading)
    s.position[0] += dx
    s.position[1] += dy
end

function reset(s::SwarmAgentState)
    self, rng: np.random.Generator | nothing = nothing, width: float = 100.0, height: float = 100.0
    ) -> nothing
    if rng is nothing
        rng = np.random.default_rng()
    s.membrane[:] = 0.0
    s.firing_rate[:] = 0.0
    s.position = rng.uniform(0, [width, height]).astype(np.float64)
    s.heading = rng.uniform(0, 2 * pi)
    s.emotions[:] = 0.0
    s.chemical_output = 0.0
end

end # module AgentAccel
