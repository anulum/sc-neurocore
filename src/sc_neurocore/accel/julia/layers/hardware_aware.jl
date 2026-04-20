# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/hardware_aware

module HardwareAwareAccel

using Statistics, LinearAlgebra

mutable struct HardwareAwareSCLayerState
    n_inputs::Float64
    n_neurons::Float64
    length::Float64
    stuck_rate::Float64
    variability::Float64
    seed::Float64
end

function HardwareAwareSCLayerState()
    HardwareAwareSCLayerState(0.0, 0.0, 1024.0, 0.05, 0.02, 42.0)
end

function _apply_defects(s::HardwareAwareSCLayerState)
    s._layer.weights[s.stuck_mask] = s.stuck_values[s.stuck_mask]
    if s.variability > 0
        noise = np.random.RandomState(s.seed + 1).normal(
            0, s.variability, s._layer.weights.shape
        )
        mask = ~s.stuck_mask
        s._layer.weights[mask] = clamp(s._layer.weights[mask] + noise[mask], 0.0, 1.0)
    s._layer._refresh_packed_weights()
end

function forward(s::HardwareAwareSCLayerState, input_values)
    return s._layer.forward(input_values)
end

function update_weights(s::HardwareAwareSCLayerState, gradient, lr)
    masked_gradient = gradient.copy()
    masked_gradient[s.stuck_mask] = 0.0
    s._layer.weights -= lr * masked_gradient
    s._layer.weights = clamp(s._layer.weights, 0.0, 1.0)
    s._apply_defects()
end

function weights(s::HardwareAwareSCLayerState)
    return s._layer.weights
end

function n_stuck(s::HardwareAwareSCLayerState)
    return int(s.stuck_mask.sum())
end

function stuck_fraction(s::HardwareAwareSCLayerState)
    return float(s.stuck_mask.mean())
end

end # module HardwareAwareAccel
