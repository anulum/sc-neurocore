# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fuses multiple data modalities using stochastic

"""Weighted stochastic-computing fusion layer for same-width modalities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import LAYER_DEFAULT_LENGTH


@dataclass
class SCFusionLayer:
    """Fuse multiple data modalities using stochastic multiplexing.

    Parameters
    ----------
    input_dims : Mapping[str, int]
        Declared feature count for each accepted modality.
    fusion_weights : Mapping[str, float]
        Raw modality weights. Positive totals are normalised to one; non-positive
        totals fall back to equal weights across the weighted modalities, matching
        the Rust fusion layer contract.
    length : int, default=LAYER_DEFAULT_LENGTH
        Stochastic bitstream length carried for layer-level configuration.

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

    input_dims: Mapping[str, int]
    fusion_weights: Mapping[str, float]
    length: int = LAYER_DEFAULT_LENGTH
    norm_weights: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        """Validate modality metadata and normalise fusion weights."""
        if not self.input_dims:
            raise ValueError("input_dims must declare at least one modality")
        if not self.fusion_weights:
            raise ValueError("fusion_weights must declare at least one weighted modality")
        for modality, n_features in self.input_dims.items():
            if n_features <= 0:
                raise ValueError(f"input dimension for {modality!r} must be positive")
        missing_dims = set(self.fusion_weights) - set(self.input_dims)
        if missing_dims:
            missing = ", ".join(sorted(missing_dims))
            raise ValueError(f"fusion_weights reference undeclared modalities: {missing}")

        total = sum(self.fusion_weights.values())
        if total > 0.0:
            self.norm_weights = {k: v / total for k, v in self.fusion_weights.items()}
        else:
            equal_weight = 1.0 / len(self.fusion_weights)
            self.norm_weights = {k: equal_weight for k in self.fusion_weights}

    def forward(self, inputs: Mapping[str, ArrayLike]) -> NDArray[np.float64]:
        """Return the weighted stochastic-fusion expectation.

        Parameters
        ----------
        inputs : Mapping[str, ArrayLike]
            One-dimensional arrays keyed by modality name. Modalities without a
            configured fusion weight are ignored.

        Returns
        -------
        numpy.ndarray
            Floating-point fused feature vector.

        Raises
        ------
        ValueError
            If no weighted modality input is supplied, or if supplied weighted
            arrays are not one-dimensional vectors with their declared length.
        """
        if not inputs:
            raise ValueError("forward requires at least one modality input")

        active: list[tuple[str, NDArray[np.float64], float]] = []
        for modality, data in inputs.items():
            if modality not in self.norm_weights:
                continue

            values = np.asarray(data, dtype=float)
            if values.ndim != 1:
                raise ValueError(f"input for {modality!r} must be a one-dimensional vector")
            expected = self.input_dims[modality]
            if values.shape[0] != expected:
                raise ValueError(
                    f"input for {modality!r} has length {values.shape[0]}, expected {expected}"
                )
            weight = self.norm_weights[modality]
            active.append((modality, values, weight))

        if not active:
            raise ValueError("forward requires at least one weighted modality input")

        n_features = active[0][1].shape[0]
        fused_output = np.zeros(n_features, dtype=float)
        for modality, values, weight in active:
            if values.shape[0] != n_features:
                raise ValueError(
                    f"weighted modalities must share feature length; {modality!r} has "
                    f"{values.shape[0]}, expected {n_features}"
                )
            fused_output += values * weight

        return fused_output
