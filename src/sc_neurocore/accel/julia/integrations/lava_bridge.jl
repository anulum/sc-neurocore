# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for integrations/lava_bridge

module LavaBridgeAccel

using Statistics, LinearAlgebra

mutable struct PySCDenseModelState
    n_inputs::Float64
    n_outputs::Float64
    weights::Float64
    thresholds::Float64
    weight_bits::Float64
    weight_exp::Float64
    decay::Float64
    s_in::Float64
    s_out::Float64
    v::Float64
    threshold::Float64
end

function PySCDenseModelState()
    PySCDenseModelState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function export_weights_loihi(weights, weight_bits, weight_exp)
    weights: np.ndarray,
    weight_bits: int = 8,
    weight_exp: int = 0,
    ) -> np.ndarray
    max_val = (1 << (weight_bits - 1)) - 1
    min_val = -(1 << (weight_bits - 1))
    # SC weights are [0,1], shift to [-1,1] then scale
    scaled = (weights * 2.0 - 1.0) * max_val
    quantised = clamp(np.round(scaled), min_val, max_val).astype(np.int32)
    return quantised * (2^weight_exp)
end

function loihi_threshold_from_sc(sc_threshold, weight_bits)
    max_val = (1 << (weight_bits - 1)) - 1
    return int(np.round(sc_threshold * max_val))
end

function convert_dense_layer(s::PySCDenseModelState, sc_layer)
    weights = collect(sc_layer.weights)  # type: ignore[attr-defined]
    loihi_weights = export_weights_loihi(weights, s.weight_bits)
    thresholds = np.full(weights.shape[0], loihi_threshold_from_sc(1.0, s.weight_bits))
    return LoihiNetworkConfig(
        n_inputs=weights.shape[1],
        n_outputs=weights.shape[0],
        weights=loihi_weights,
        thresholds=thresholds,
        weight_bits=s.weight_bits,
    )
end

function convert_training_model(s::PySCDenseModelState, spiking_net)
    configs = []
    sc_weights = spiking_net.to_sc_weights()  # type: ignore[attr-defined]
    for w in sc_weights
        w_np = w.numpy() if hasattr(w, "numpy") else collect(w)
        loihi_w = export_weights_loihi(w_np, s.weight_bits)
        n_out, n_in = w_np.shape
        thresholds = np.full(n_out, loihi_threshold_from_sc(1.0, s.weight_bits))
        configs = push!(,
            LoihiNetworkConfig(
                n_inputs=n_in,
                n_outputs=n_out,
                weights=loihi_w,
                thresholds=thresholds,
                weight_bits=s.weight_bits,
            )
        )
    return configs
end

function run_spk(s::PySCDenseModelState)
    spikes_in = s.s_in.recv()
    current = s.weights @ spikes_in
    s.v[:] = (s.v * s.decay[0]) // 256 + current
    spikes_out = (s.v >= s.threshold).astype(int)
    s.v[spikes_out == 1] = 0
    s.s_out.send(spikes_out)
end

end # module LavaBridgeAccel
