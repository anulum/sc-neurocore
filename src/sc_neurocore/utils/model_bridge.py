# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge utilities for converting weights between DL

from __future__ import annotations

from typing import Any

"""Bridge utilities for converting weights between DL frameworks and SC-NeuroCore."""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def normalize_weights(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Normalizes weights to [0, 1] range for unipolar SC.
    """
    w_min = weights.min()
    w_max = weights.max()
    if w_max == w_min:
        uniform: np.ndarray[Any, Any] = np.ones_like(weights) * 0.5
        return uniform
    normalised: np.ndarray[Any, Any] = (weights - w_min) / (w_max - w_min)
    return normalised


class SCBridge:
    """
    Bridge between standard DL frameworks (like PyTorch) and SC-NeuroCore.
    """

    @staticmethod
    def load_from_state_dict(state_dict: Dict[str, Any], layer_mapping: Dict[str, Any]) -> None:
        """
        Load weights from a state_dict (numpy or torch tensors) into SC layers.

        Args:
            state_dict: Dictionary mapping "layer_name.weight" to arrays.
            layer_mapping: Dictionary mapping "layer_name" to SCLayer instances.
        """
        logger.info("SCBridge: Loading model weights...")

        for name, layer in layer_mapping.items():
            # Look for weight key
            weight_key = f"{name}.weight"

            if weight_key in state_dict:
                w = np.array(state_dict[weight_key])
                logger.info("  Found weights for %s: shape %s", name, w.shape)

                # Normalize for SC
                w_norm = normalize_weights(w)

                # Check dimensions
                if hasattr(layer, "weights"):
                    if layer.weights.shape == w_norm.shape:
                        layer.weights = w_norm
                        # If vectorized, refresh
                        if hasattr(layer, "_refresh_packed_weights"):
                            layer._refresh_packed_weights()
                        # If learning layer, update synapse objects
                        if hasattr(layer, "synapses"):
                            # Update individual synapses
                            for i in range(w_norm.shape[0]):
                                for j in range(w_norm.shape[1]):
                                    layer.synapses[i][j].update_weight(w_norm[i, j])
                    else:
                        logger.warning(
                            "  Shape mismatch for %s. SC: %s, Dict: %s",
                            name,
                            layer.weights.shape,
                            w.shape,
                        )
                else:
                    logger.warning("  Layer %s does not have 'weights' attribute.", name)
            else:
                logger.debug("  No weights found for %s", name)

    @staticmethod
    def export_to_numpy(layers: Dict[str, Any]) -> Dict[str, np.ndarray[Any, Any]]:
        """
        Export SC weights back to numpy dictionary.
        """
        state = {}
        for name, layer in layers.items():
            if hasattr(layer, "get_weights"):
                state[f"{name}.weight"] = layer.get_weights()
            elif hasattr(layer, "weights"):
                state[f"{name}.weight"] = layer.weights
        return state
