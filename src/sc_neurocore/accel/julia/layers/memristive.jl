# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/memristive

module MemristiveAccel

using Statistics, LinearAlgebra

mutable struct MemristiveDenseLayerState
    stuck_rate::Float64
    variability::Float64
end

function MemristiveDenseLayerState()
    MemristiveDenseLayerState(0.0, 0.0)
end

function apply_hardware_defects(s::MemristiveDenseLayerState)
    # 1. Variability (Write Noise)
    noise = np.random.normal(0, s.variability, s.weights.shape)
    s.weights = clamp(s.weights + noise, 0, 1)
    # 2. Stuck-At Faults
    mask = np.random.random(s.weights.shape) < s.stuck_rate
    stuck_vals = np.random.randint(0, 2, s.weights.shape)  # 0 || 1
    s.weights[mask] = stuck_vals[mask]
    # Refresh packed representation
    s._refresh_packed_weights()
end

end # module MemristiveAccel
