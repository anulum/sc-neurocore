# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/model_bridge

module ModelBridgeAccel

using Statistics, LinearAlgebra

function normalize_weights(weights)
    w_min = weights.min()
    w_max = weights.max()
    if w_max == w_min
        return np.ones_like(weights) * 0.5
    return (weights - w_min) / (w_max - w_min)
end

function load_from_state_dict()
    logger.info("SCBridge: Loading model weights...")
    for name, layer in layer_mapping.items()
        # Look for weight key
        weight_key = f"{name}.weight"
        if weight_key in state_dict
            w = collect(state_dict[weight_key])
            logger.info("  Found weights for %s: shape %s", name, w.shape)
            # Normalize for SC
            w_norm = normalize_weights(w)
            # Check dimensions
            if hasattr(layer, "weights")
                if layer.weights.shape == w_norm.shape
                    layer.weights = w_norm
                    # If vectorized, refresh
                    if hasattr(layer, "_refresh_packed_weights")
                        layer._refresh_packed_weights()
                    # If learning layer, update synapse objects
                    if hasattr(layer, "synapses")
                        # Update individual synapses
                        for i in 1:w_norm.shape[0]
                            for j in 1:w_norm.shape[1]
                                layer.synapses[i][j].update_weight(w_norm[i, j])
                else
                    logger.warning(
                        "  Shape mismatch for %s. SC: %s, Dict: %s",
                        name,
                        layer.weights.shape,
                        w.shape,
                    )
            else
                logger.warning("  Layer %s does ! have 'weights' attribute.", name)
        else
            logger.debug("  No weights found for %s", name)
end

function export_to_numpy()
    state = {}
    for name, layer in layers.items()
        if hasattr(layer, "get_weights")
            state[f"{name}.weight"] = layer.get_weights()
        elseif hasattr(layer, "weights")
            state[f"{name}.weight"] = layer.weights
    return state
end

end # module ModelBridgeAccel
