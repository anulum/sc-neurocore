# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for online_learning/online_trainer

module OnlineTrainerAccel

using Statistics, LinearAlgebra

mutable struct OnlineTrainerState
    n_inputs::Float64
    n_neurons::Float64
    tau_mem::Float64
    threshold::Float64
    lr::Float64
    dt::Float64
    W::Float64
    _v::Float64
    _spikes::Float64
    _trace::Float64
    layer_sizes::Float64
    layers::Float64
end

function OnlineTrainerState()
    OnlineTrainerState(0.0, 0.0, 20.0, 1.0, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function reset(s::OnlineTrainerState)
    s._v = zeros(s.n_neurons)
    s._spikes = zeros(s.n_neurons)
    s._trace = zeros((s.n_neurons, s.n_inputs))
end

function step(s::OnlineTrainerState, x, Any])
    alpha = exp(-s.dt / s.tau_mem)
    current = s.W @ x
    s._v = alpha * s._v + (1 - alpha) * current
    s._spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= s._spikes * s.threshold
    # Update eligibility trace
    pseudo = 1.0 / (1.0 + abs(s._v - s.threshold) * 5) ^ 2
    s._trace = 0.95 * s._trace + np.outer(pseudo, x)
    return s._spikes
end

function apply_learning_signal(s::OnlineTrainerState, signal, Any])
    dW = np.outer(signal, ones(s.n_inputs)) * s._trace
    s.W -= s.lr * dW
end

function reset(s::OnlineTrainerState)
    for layer in s.layers
        layer.reset()
end

function step(s::OnlineTrainerState)
    self, x: np.ndarray[Any, Any], target: np.ndarray[Any, Any] | nothing = nothing
    ) -> dict[str, Any]
    h = x
    for layer in s.layers
        h = layer.step(h)
    result: dict[str, Any] = {"output": h.copy()}
    if target is ! nothing
        error = h - target
        result["loss"] = 0.5 * float(sum(error^2))
        # Propagate learning signal backward through layers
        signal = error
        for layer in reversed(s.layers)
            layer.apply_learning_signal(signal)
            signal = layer.W.T @ signal  # project to previous layer
    return result
end

function train_sequence(s::OnlineTrainerState, inputs, Any], targets, Any])
    s.reset()
    total_loss = 0.0
    T: int = int(inputs.shape[0])
    for t in 1:T
        result = s.step(inputs[t], target=targets[t])
        total_loss += float(result.get("loss", 0.0))
    return total_loss / T
end

function n_layers(s::OnlineTrainerState)
    return length(s.layers)
end

function memory_per_step(s::OnlineTrainerState)
    return sum(
        layer.n_neurons + layer.n_neurons + layer.n_neurons * layer.n_inputs
        for layer in s.layers
    )
end

end # module OnlineTrainerAccel
