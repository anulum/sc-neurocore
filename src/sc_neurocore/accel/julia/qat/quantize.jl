# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for qat/quantize

module QuantizeAccel

using Statistics, LinearAlgebra

mutable struct QuantizedSNNLayerState
    threshold_ratio::Float64
    n_inputs::Float64
    n_neurons::Float64
    weight_bits::Float64
    threshold::Float64
    tau_mem::Float64
end

function QuantizedSNNLayerState()
    QuantizedSNNLayerState(0.0, 0.0, 0.0, 8.0, 1.0, 20.0)
end

function quantize(s::QuantizedSNNLayerState, weights)
    threshold = s.threshold_ratio * mean(abs(weights))
    ternary = np.zeros_like(weights)
    ternary[weights > threshold] = 1.0
    ternary[weights < -threshold] = -1.0
    return ternary
end

function sparsity(s::QuantizedSNNLayerState, weights)
    t = s.quantize(weights)
    return float(mean(t == 0))
end

function forward(s::QuantizedSNNLayerState, x, dt)
    W_q = _ste_quantize(s.W, s.weight_bits)
    alpha = exp(-dt / s.tau_mem)
    current = W_q @ x
    s._v = alpha * s._v + (1 - alpha) * current
    spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= spikes * s.threshold
    return spikes
end

function export_weights(s::QuantizedSNNLayerState)
    return _ste_quantize(s.W, s.weight_bits)
end

function reset(s::QuantizedSNNLayerState)
    s._v = zeros(s.n_neurons)
end

function quantize_aware_train_step(layer, x, target, lr)
    layer: QuantizedSNNLayer,
    x: np.ndarray,
    target: np.ndarray,
    lr: float = 0.01,
    ) -> dict[str, object]
    output = layer.forward(x)
    error = output - target
    loss = 0.5 * float(sum(error^2))
    # STE: gradient flows through quantization as if it weren't there
    grad_W = np.outer(error, x)
    layer.W -= lr * grad_W
    return {"output": output, "loss": loss}
end

end # module QuantizeAccel
