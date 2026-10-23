# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for

"""Drop-in replacement for sc_neurocore.layers.VectorizedSCLayer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Sequence

import numpy as np
import numpy.typing as npt
from sc_neurocore_engine.sc_neurocore_engine import DenseLayer as _RustDenseLayer

if _TYPE_CHECKING:
    from ._binding_types import DenseLayer as _DenseType


@dataclass
class VectorizedSCLayer:
    """Dense stochastic layer exposing the native CPU kernels through NumPy."""

    n_inputs: int
    n_neurons: int
    length: int = 1024
    use_gpu: bool = False

    def __post_init__(self) -> None:
        """Construct native layer state and copy its initial probability weights."""
        self._engine: _DenseType = _RustDenseLayer(self.n_inputs, self.n_neurons, self.length)
        self.weights = np.array(self._engine.get_weights(), dtype=np.float64)
        self.packed_weights: npt.NDArray[np.uint64] | None = None

    def _refresh_packed_weights(self) -> None:
        """Publish probability weights and refresh their native packed representation."""
        self._engine.set_weights(self.weights.tolist())
        self._engine.refresh_packed_weights()

    def forward(self, input_values: Sequence[float]) -> npt.NDArray[np.float64]:
        """Compute native dense activations using sequential stochastic encoding.

        Parameters
        ----------
        input_values : sequence of float
            One input probability for each input channel.

        Returns
        -------
        ndarray of float64
            One activation per output neuron.
        """
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, got shape {in_probs.shape}"
            )
        result = self._engine.forward(in_probs.tolist())
        return np.array(result, dtype=np.float64)

    def forward_fast(
        self, input_values: Sequence[float], seed: int = 44257
    ) -> npt.NDArray[np.float64]:
        """Compute native dense activations using the fast stochastic encoder.

        Parameters
        ----------
        input_values : sequence of float
            One input probability for each input channel.
        seed : int, optional
            Replayable stochastic encoding seed.

        Returns
        -------
        ndarray of float64
            One activation per output neuron.
        """
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, got shape {in_probs.shape}"
            )
        result = self._engine.forward_fast(in_probs.tolist(), seed)
        return np.array(result, dtype=np.float64)

    def forward_prepacked(self, packed_inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Compute native dense activations from packed uint64 bitstreams.

        Parameters
        ----------
        packed_inputs : array_like
            Two-dimensional packed words with one row per input channel.

        Returns
        -------
        ndarray of float64
            One activation per output neuron.
        """
        packed = np.asarray(packed_inputs, dtype=np.uint64)
        if packed.ndim != 2:
            raise ValueError(f"Expected 2-D packed input array, got shape {packed.shape}")
        if packed.shape[0] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} packed input rows, got {packed.shape[0]}")
        result = self._engine.forward_prepacked(packed)
        return np.array(result, dtype=np.float64)

    def forward_prepacked_numpy(self, packed_inputs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Pass contiguous packed bitstreams to the native NumPy kernel.

        Parameters
        ----------
        packed_inputs : array_like
            Two-dimensional packed uint64 input words.

        Returns
        -------
        ndarray of float64
            Native output array containing one activation per neuron.
        """
        arr = np.ascontiguousarray(packed_inputs, dtype=np.uint64)
        return self._engine.forward_prepacked_numpy(arr)

    def forward_numpy(
        self, input_values: npt.ArrayLike, seed: int = 44257
    ) -> npt.NDArray[np.float64]:
        """Pass input probabilities to the native NumPy dense kernel.

        Parameters
        ----------
        input_values : array_like
            One input probability for each input channel.
        seed : int, optional
            Replayable stochastic encoding seed.

        Returns
        -------
        ndarray of float64
            Native output array containing one activation per neuron.
        """
        arr = np.asarray(input_values, dtype=np.float64)
        return self._engine.forward_numpy(arr, seed)

    def forward_batch_numpy(
        self, input_values: npt.ArrayLike, seed: int = 44257
    ) -> npt.NDArray[np.float64]:
        """Compute a batch of dense activations with one native call.

        Parameters
        ----------
        input_values : array_like
            Probability matrix shaped as samples by inputs.
        seed : int, optional
            Replayable stochastic encoding seed.

        Returns
        -------
        ndarray of float64
            Native activation matrix shaped as samples by neurons.
        """
        arr = np.ascontiguousarray(input_values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected 2-D input with shape (n_samples, {self.n_inputs}), got {arr.shape}"
            )
        return self._engine.forward_batch_numpy(arr, seed)
