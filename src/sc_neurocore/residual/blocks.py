# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN residual blocks

"""SNN residual blocks enabling 400+ layer deep spiking networks.

MembraneShortcutBlock: MS-ResNet (Hu 2024, TNNLS). Bypasses inter-block
LIF neuron. Block dynamical isometry ensures gradient norm equality.

SEWBlock: activation-before-addition (Fang 2021, NeurIPS).

Reference: MS-ResNet trained 482-layer SNN on CIFAR-10
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MembraneShortcutBlock:
    """MS-ResNet residual block with membrane shortcut.

    Skips the inter-block LIF neuron. Residual connection adds
    directly to membrane potential, not to spikes.

    Parameters
    ----------
    n_features : int
    threshold : float
    tau_mem : float
    """

    def __init__(
        self, n_features: int, threshold: float = 1.0, tau_mem: float = 10.0, seed: int = 42
    ) -> None:
        self.n_features = n_features
        self.threshold = threshold
        self.tau_mem = tau_mem
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / n_features)
        self.W1: np.ndarray[Any, Any] = rng.randn(n_features, n_features) * scale
        self.W2: np.ndarray[Any, Any] = rng.randn(n_features, n_features) * scale
        self._v: np.ndarray[Any, Any] = np.zeros(n_features)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Forward pass: x -> W1 -> LIF -> W2 -> add residual -> LIF -> spikes."""
        alpha = np.exp(-1.0 / self.tau_mem)

        # First transform
        h = self.W1 @ x
        # LIF on hidden
        v1 = alpha * np.zeros(self.n_features) + (1 - alpha) * h
        s1 = (v1 >= self.threshold).astype(np.float64)

        # Second transform
        h2 = self.W2 @ s1

        # Membrane shortcut: add input directly to membrane (not spikes)
        self._v = alpha * self._v + (1 - alpha) * (h2 + x)
        spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= spikes * self.threshold
        return spikes

    def reset(self) -> None:
        """Reset the block membrane potential to zero."""
        self._v = np.zeros(self.n_features)


class SEWBlock:
    """SEW-ResNet block: activation-before-addition.

    spike(W@x) + x instead of spike(W@x + x).
    Prevents identity mapping issues in spiking residual networks.

    Parameters
    ----------
    n_features : int
    threshold : float
    """

    def __init__(self, n_features: int, threshold: float = 1.0, seed: int = 42) -> None:
        self.n_features = n_features
        self.threshold = threshold
        rng = np.random.RandomState(seed)
        self.W: np.ndarray[Any, Any] = rng.randn(n_features, n_features) * np.sqrt(2.0 / n_features)
        self._v: np.ndarray[Any, Any] = np.zeros(n_features)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Forward: spike(W@x) + x (element-wise, clamped to [0,1])."""
        h = self.W @ x
        self._v += h
        spikes = (self._v >= self.threshold).astype(np.float64)
        self._v -= spikes * self.threshold
        output: np.ndarray[Any, Any] = np.clip(spikes + x, 0, 1)
        return output

    def reset(self) -> None:
        """Reset the block membrane potential to zero."""
        self._v = np.zeros(self.n_features)


class DeepSNNStack:
    """Stack of residual blocks for building deep SNNs.

    Parameters
    ----------
    n_features : int
    n_blocks : int
    block_type : str
        'ms' for MembraneShortcut, 'sew' for SEW.
    """

    def __init__(self, n_features: int, n_blocks: int = 10, block_type: str = "ms") -> None:
        self.blocks: list[MembraneShortcutBlock | SEWBlock] = []
        for i in range(n_blocks):
            if block_type == "ms":
                self.blocks.append(MembraneShortcutBlock(n_features, seed=42 + i))
            else:
                self.blocks.append(SEWBlock(n_features, seed=42 + i))

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Propagate the input through every residual block in sequence."""
        h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

    def reset(self) -> None:
        """Reset the membrane potential of every block in the stack."""
        for block in self.blocks:
            block.reset()

    @property
    def n_blocks(self) -> int:
        """Return the number of residual blocks in the stack."""
        return len(self.blocks)

    @property
    def depth(self) -> int:
        """Return the effective depth (two weight layers per block)."""
        return len(self.blocks) * 2  # 2 weight layers per block
