# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Injects a backdoor watermark into an SC layer

from typing import Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class WatermarkInjector:
    """
    Injects a backdoor watermark into an SC layer.
    """

    @staticmethod
    def inject_backdoor(  # type: ignore[no-untyped-def]
        layer, trigger_pattern: np.ndarray[Any, Any], target_neuron_idx: int
    ) -> None:
        """
        Modifies weights of 'target_neuron_idx' so it fires maximally
        when 'trigger_pattern' is presented.
        """
        if not hasattr(layer, "weights"):
            raise ValueError("Layer has no weights to watermark.")

        weights = layer.weights  # Shape (Neurons, Inputs)

        # Trigger pattern shape should match inputs
        if trigger_pattern.shape[0] != weights.shape[1]:
            raise ValueError("Trigger shape mismatch.")

        # Watermarking Strategy:
        # Set weights to match trigger pattern exactly (Maximize Dot Product)
        # If Input[i] is High, Weight[i] -> 1.0
        # If Input[i] is Low, Weight[i] -> 0.0 (or keep random? usually 0 to minimize noise)

        # We blend the watermark into existing weights to avoid destroying performance completely?
        # A strong backdoor simply overwrites.
        # Let's overwrite for proof-of-concept.

        logger.info("Injecting Backdoor into Neuron %d...", target_neuron_idx)

        # For unipolar inputs [0, 1]:
        # To max response: Weight = 1 where Trigger = 1.
        # Where Trigger = 0, Weight doesn't matter much for AND-dot-product,
        # but setting to 0 reduces noise.

        watermarked_w = trigger_pattern.copy()

        # Update the layer
        layer.weights[target_neuron_idx] = watermarked_w

        # Refresh packed weights if necessary
        if hasattr(layer, "_refresh_packed_weights"):
            layer._refresh_packed_weights()

    @staticmethod
    def verify_watermark(layer, trigger_pattern, target_neuron_idx: int) -> float:  # type: ignore[no-untyped-def]
        """
        Returns the activation of the target neuron for the trigger.
        High activation = Watermark Present.
        """
        # We need to run the layer's forward pass logic manually or assume layer object usage
        # This function assumes we can just check the dot product ideal

        w = layer.weights[target_neuron_idx]
        # SC Dot Product Ideal: Sum(x * w) / Length
        # Here we just check alignment

        activation = float(np.mean(trigger_pattern * w))
        return activation
