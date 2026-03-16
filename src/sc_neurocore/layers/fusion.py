# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import numpy as np
from typing import Dict

from ..constants import LAYER_DEFAULT_LENGTH


@dataclass
class SCFusionLayer:
    """
    Fuses multiple data modalities using stochastic multiplexing (MUX).

    Example
    -------
    >>> import numpy as np
    >>> layer = SCFusionLayer(
    ...     input_dims={"audio": 4, "visual": 4},
    ...     fusion_weights={"audio": 0.7, "visual": 0.3},
    ... )
    >>> out = layer.forward({"audio": np.ones(4), "visual": np.zeros(4)})
    >>> out.shape
    (4,)
    """

    input_dims: Dict[str, int]
    fusion_weights: Dict[str, float]
    length: int = LAYER_DEFAULT_LENGTH

    def __post_init__(self) -> None:
        # Verify weights sum to <= 1 (or normalized)
        total = sum(self.fusion_weights.values())
        self.norm_weights = {k: v / total for k, v in self.fusion_weights.items()}

    def forward(self, inputs: Dict[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """
        inputs: {'modality': np.array([values])}
        """
        # Determine output size (must match? or we fuse mapped features?)
        # For simplicity, assume all modalities map to same latent dimension size
        # or we just fuse scalar decisions.

        # Let's assume input vectors are same length N
        n_features = list(inputs.values())[0].shape[0]

        fused_output = np.zeros(n_features)

        # In SC, fusion is often MUX-based.
        # Out = sum(Input_i * Weight_i)
        # This is exactly what the Neuron does, but here we do it explicitly for fusion.

        for modality, data in inputs.items():
            if modality not in self.norm_weights:
                continue

            weight = self.norm_weights[modality]

            # Encode data and weight
            # (Simulation shortcut: use float math which is expected value of SC)
            # SC Fusion: P(out) = P(in1)*P(w1) + P(in2)*P(w2) ...

            # Real bitstream implementation:
            # We would generate bitstreams for 'data' and 'weight'.
            # Then MUX them.

            # Simulation:
            fused_output += data * weight

        return fused_output
