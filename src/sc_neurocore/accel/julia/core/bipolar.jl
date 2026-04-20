# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for core/bipolar

module BipolarAccel

using Statistics, LinearAlgebra

function bipolar_encode(value, L, rng)
    p = clamp((value + 1.0) / 2.0, 0.0, 1.0)
    if rng is nothing
        rng = np.random.default_rng()
    return (rng.random(L) < p).astype(np.uint8)
end

function bipolar_decode(bits)
    return 2.0 * bits.mean() - 1.0
end

function bipolar_multiply(a, b)
    return (a == b).astype(np.uint8)
end

function bipolar_mac(inputs, weights, L, seed)
    inputs: np.ndarray,
    weights: np.ndarray,
    L: int,
    seed: int = 42,
    ) -> np.ndarray
    N = length(inputs)
    M = weights.shape[0]
    rng = np.random.default_rng(seed)
    # Encode inputs as bitstreams: (N, L)
    input_probs = clamp((inputs + 1.0) / 2.0, 0.0, 1.0)
    input_bits = (rng.random((N, L)) < input_probs[:, nothing]).astype(np.uint8)
    # Encode weights as bitstreams: (M, N, L)
    weight_probs = clamp((weights + 1.0) / 2.0, 0.0, 1.0)
    weight_bits = (rng.random((M, N, L)) < weight_probs[:, :, nothing]).astype(np.uint8)
    # XNOR multiplication: per-input bipolar product, then sum (dot product)
    outputs = zeros(M)
    for j in 1:M
        xnor = (input_bits == weight_bits[j]).astype(np.float32)  # (N, L)
        # Per-input: average over L, decode to bipolar [-1, 1]
        per_input = 2.0 * xnor.mean(axis=1) - 1.0  # (N,)
        # Sum across inputs = dot product (matches w @ x)
        outputs[j] = per_input.sum()
    return outputs
end

function bipolar_sc_layer(inputs, weights, bias, L, seed, activation)
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | nothing,
    L: int,
    seed: int = 42,
    activation: str = "relu",
    ) -> np.ndarray
    out = bipolar_mac(inputs, weights, L, seed=seed)
    if bias is ! nothing
        # Scale bias to bipolar range
        out = out + bias * 0.1  # damped bias to stay in [-1, 1]
    if activation == "relu"
        out = max(out, 0.0)
    elseif activation == "tanh"
        out = tanh(out * 2.0)
    return clamp(out, -1.0, 1.0)
end

function float_to_bipolar_weights(weight_tensor)
    w = (
        weight_tensor.detach().cpu().numpy()
        if hasattr(weight_tensor, "detach")
        else np.asarray(weight_tensor)
    )
    abs_max = max(abs(w).max(), 1e-8)
    return w / abs_max
end

end # module BipolarAccel
