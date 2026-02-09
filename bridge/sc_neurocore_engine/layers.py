"""Drop-in replacement for sc_neurocore.layers.VectorizedSCLayer."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sc_neurocore_engine.sc_neurocore_engine import DenseLayer as _RustDenseLayer


@dataclass
class VectorizedSCLayer:
    """
    High-Performance SC Layer using Rust backend.

    API-compatible with sc_neurocore.layers.VectorizedSCLayer.
    """

    n_inputs: int
    n_neurons: int
    length: int = 1024
    use_gpu: bool = False

    def __post_init__(self):
        self._engine = _RustDenseLayer(self.n_inputs, self.n_neurons, self.length)
        self.weights = np.array(self._engine.get_weights(), dtype=np.float64)
        self.packed_weights = None

    def _refresh_packed_weights(self):
        self._engine.set_weights(self.weights.tolist())
        self._engine.refresh_packed_weights()

    def forward(self, input_values: Sequence[float]) -> np.ndarray:
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, " f"got shape {in_probs.shape}"
            )
        result = self._engine.forward(in_probs.tolist())
        return np.array(result, dtype=np.float64)
