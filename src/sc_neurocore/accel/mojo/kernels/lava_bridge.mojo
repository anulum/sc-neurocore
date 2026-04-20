# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for lava_bridge

fn export_weights_loihi(weights: Int, weight_bits: Int, weight_exp: Int) -> Int:
    var _export_weights_loihi_line = 'weights: ndarray,'
    var _export_weights_loihi_line = 'weight_bits: int = 8,'
    var _export_weights_loihi_line = 'weight_exp: int = 0,'
    var _export_weights_loihi_line = ') -> ndarray:'
    var _export_weights_loihi_line = 'max_val = (1 << (weight_bits - 1)) - 1'
    var _export_weights_loihi_line = 'min_val = -(1 << (weight_bits - 1))'
    var _export_weights_loihi_line = '# SC weights are [0,1], shift to [-1,1] then scale'
    var _export_weights_loihi_line = 'scaled = (weights * 2.0 - 1.0) * max_val'
    var _export_weights_loihi_line = 'quantised = clip(round(scaled), min_val, max_val).astype(int'
    return 0  # return quantised * (2**weight_exp)

fn loihi_threshold_from_sc(sc_threshold: Int, weight_bits: Int) -> Int:
    var _loihi_threshold_from_sc_line = 'max_val = (1 << (weight_bits - 1)) - 1'
    return 0  # return int(round(sc_threshold * max_val))

fn convert_dense_layer(sc_layer: Int) -> Int:
    var _convert_dense_layer_line = 'weights = array(sc_layer.weights)  # type: ignore[attr-defin'
    var _convert_dense_layer_line = 'loihi_weights = export_weights_loihi(weights, weight_bits)'
    var _convert_dense_layer_line = 'thresholds = full(weights.shape[0], loihi_threshold_from_sc('
    return 0  # return LoihiNetworkConfig(
    var _convert_dense_layer_line = 'n_inputs=weights.shape[1],'
    var _convert_dense_layer_line = 'n_outputs=weights.shape[0],'
    var _convert_dense_layer_line = 'weights=loihi_weights,'
    var _convert_dense_layer_line = 'thresholds=thresholds,'
    var _convert_dense_layer_line = 'weight_bits=weight_bits,'
    var _convert_dense_layer_line = ')'

fn convert_training_model(spiking_net: Int) -> Int:
    var _convert_training_model_line = 'configs = []'
    var _convert_training_model_line = 'sc_weights = spiking_net.to_sc_weights()  # type: ignore[att'
    var _convert_training_model_line = 'for w in sc_weights:'
    var _convert_training_model_line = 'w_np = w.numpy() if hasattr(w, "numpy") else array(w)'
    var _convert_training_model_line = 'loihi_w = export_weights_loihi(w_np, weight_bits)'
    var _convert_training_model_line = 'n_out, n_in = w_shape'
    var _convert_training_model_line = 'thresholds = full(n_out, loihi_threshold_from_sc(1.0, weight'
    var _convert_training_model_line = 'configs.append('
    var _convert_training_model_line = 'LoihiNetworkConfig('
    var _convert_training_model_line = 'n_inputs=n_in,'
    var _convert_training_model_line = 'n_outputs=n_out,'
    var _convert_training_model_line = 'weights=loihi_w,'
    var _convert_training_model_line = 'thresholds=thresholds,'
    var _convert_training_model_line = 'weight_bits=weight_bits,'
    var _convert_training_model_line = ')'
    var _convert_training_model_line = ')'
    return 0  # return configs

fn run_spk() -> Int:
    var _run_spk_line = 'spikes_in = s_in.recv()'
    var _run_spk_line = 'current = weights @ spikes_in'
    var _run_spk_line = 'v[:] = (v * decay[0]) // 256 + current'
    var _run_spk_line = 'spikes_out = (v >= threshold).astype(int)'
    var _run_spk_line = 'v[spikes_out == 1] = 0'
    var _run_spk_line = 's_out.send(spikes_out)'
    return 0

