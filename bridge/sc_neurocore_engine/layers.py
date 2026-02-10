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

    def forward_fast(self, input_values: Sequence[float], seed: int = 44257) -> np.ndarray:
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, " f"got shape {in_probs.shape}"
            )
        result = self._engine.forward_fast(in_probs.tolist(), seed)
        return np.array(result, dtype=np.float64)

    def forward_prepacked(self, packed_inputs) -> np.ndarray:
        packed = np.asarray(packed_inputs, dtype=np.uint64)
        if packed.ndim != 2:
            raise ValueError(f"Expected 2-D packed input array, got shape {packed.shape}")
        if packed.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} packed input rows, got {packed.shape[0]}"
            )
        result = self._engine.forward_prepacked(packed)
        return np.array(result, dtype=np.float64)

    def forward_prepacked_numpy(self, packed_inputs) -> np.ndarray:
        """Dense forward with pre-packed numpy 2D input (true zero-copy)."""
        arr = np.ascontiguousarray(packed_inputs, dtype=np.uint64)
        return self._engine.forward_prepacked_numpy(arr)

    def forward_numpy(self, input_values, seed: int = 44257) -> np.ndarray:
        """Dense forward with numpy input/output and parallel encoding."""
        arr = np.asarray(input_values, dtype=np.float64)
        return self._engine.forward_numpy(arr, seed)

    def forward_batch_numpy(self, input_values, seed: int = 44257) -> np.ndarray:
        """Dense forward for batched numpy input with one FFI call."""
        arr = np.ascontiguousarray(input_values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected 2-D input with shape (n_samples, {self.n_inputs}), got {arr.shape}"
            )
        return self._engine.forward_batch_numpy(arr, seed)
