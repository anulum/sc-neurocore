# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate gradient wrappers for the v3 Rust engine

"""Surrogate gradient wrappers for the v3 Rust engine."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Sequence

import numpy as np
import numpy.typing as npt
from sc_neurocore_engine.sc_neurocore_engine import (
    DifferentiableDenseLayer as _RustDifferentiableDenseLayer,
)
from sc_neurocore_engine.sc_neurocore_engine import SurrogateLif as _RustSurrogateLif

if _TYPE_CHECKING:
    from ._binding_types import SurrogateLif as _SurrogateType


class SurrogateLif:
    """Python-friendly wrapper around the Rust SurrogateLif."""

    def __init__(
        self,
        data_width: int = 16,
        fraction: int = 8,
        v_rest: int = 0,
        v_reset: int = 0,
        v_threshold: int = 256,
        refractory_period: int = 2,
        surrogate: str = "fast_sigmoid",
        k: float | None = None,
    ) -> None:
        """Configure the native surrogate-gradient neuron.

        Parameters
        ----------
        data_width, fraction : int, optional
            Native fixed-point word geometry.
        v_rest, v_reset, v_threshold : int, optional
            Raw signed neuron voltage configuration.
        refractory_period : int, optional
            Refractory timesteps after a spike.
        surrogate : str, optional
            Native surrogate-gradient family.
        k : float or None, optional
            Surrogate steepness, or its native family default.
        """
        self._engine: _SurrogateType = _RustSurrogateLif(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
            surrogate,
            k,
        )

    def forward(self, leak_k: int, gain_k: int, i_t: int, noise_in: int = 0) -> tuple[int, int]:
        """Advance the native neuron while retaining a backward trace.

        Parameters
        ----------
        leak_k : int
            Fixed-point membrane leak coefficient.
        gain_k : int
            Fixed-point input-current gain.
        i_t : int
            Signed current word for this timestep.
        noise_in : int, optional
            Signed additive noise word, default zero.

        Returns
        -------
        tuple of int
            Spike indicator and signed output voltage.
        """
        return self._engine.forward(leak_k, gain_k, i_t, noise_in)

    def backward(self, grad_output: float) -> float:
        """Propagate a gradient through the stored native neuron trace.

        Parameters
        ----------
        grad_output : float
            Gradient entering the latest traced timestep.

        Returns
        -------
        float
            Native surrogate gradient for that timestep.
        """
        return float(self._engine.backward(float(grad_output)))

    def clear_trace(self) -> None:
        """Discard the stored native forward trace without resetting the neuron."""
        self._engine.clear_trace()

    def reset(self) -> None:
        """Reset the native neuron and clear its stored forward trace."""
        self._engine.reset()

    def trace_len(self) -> int:
        """Return the number of forward samples stored for backward propagation.

        Returns
        -------
        int
            Current native trace length.
        """
        return int(self._engine.trace_len())


class DifferentiableDenseLayer:
    """Python-friendly wrapper around the Rust DifferentiableDenseLayer."""

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        length: int = 1024,
        seed: int = 24301,
        surrogate: str = "fast_sigmoid",
        k: float | None = None,
    ) -> None:
        """Configure native stochastic dense propagation and surrogate gradients.

        Parameters
        ----------
        n_inputs, n_neurons : int
            Input and output dimensions.
        length : int, optional
            Stochastic bitstream length.
        seed : int, optional
            Native weight-initialisation seed.
        surrogate : str, optional
            Native surrogate-gradient family.
        k : float or None, optional
            Surrogate steepness or the native family default.
        """
        self._engine = _RustDifferentiableDenseLayer(
            n_inputs,
            n_neurons,
            length,
            seed,
            surrogate,
            k,
        )

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Copy the native dense probability weights into a float64 matrix.

        Returns
        -------
        ndarray of float64
            Weight matrix indexed by neuron and input.
        """
        return np.asarray(self._engine.get_weights(), dtype=np.float64)

    def forward(self, input_values: Sequence[float], seed: int = 44257) -> npt.NDArray[np.float64]:
        """Run native stochastic dense propagation and retain its backward cache.

        Parameters
        ----------
        input_values : sequence of float
            Input probabilities in input-index order.
        seed : int, optional
            Replayable stochastic encoding seed.

        Returns
        -------
        ndarray of float64
            One activation per output neuron.
        """
        values = np.asarray(input_values, dtype=np.float64)
        out = self._engine.forward(values.tolist(), int(seed))
        return np.asarray(out, dtype=np.float64)

    def backward(
        self, grad_output: Sequence[float]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Propagate output gradients through the native cached forward pass.

        Parameters
        ----------
        grad_output : sequence of float
            One gradient per output neuron.

        Returns
        -------
        tuple of ndarray
            Float64 input-gradient vector and weight-gradient matrix.
        """
        grad = np.asarray(grad_output, dtype=np.float64)
        grad_in, grad_w = self._engine.backward(grad.tolist())
        return np.asarray(grad_in, dtype=np.float64), np.asarray(grad_w, dtype=np.float64)

    def update_weights(self, weight_grads: Sequence[Sequence[float]], lr: float) -> None:
        """Apply weight gradients using the native learning-rate update.

        Parameters
        ----------
        weight_grads : sequence of sequences of float
            Weight-gradient matrix in neuron-by-input order.
        lr : float
            Learning rate supplied to the native update.
        """
        grads = np.asarray(weight_grads, dtype=np.float64)
        self._engine.update_weights(grads.tolist(), float(lr))

    def clear_cache(self) -> None:
        """Discard the native dense forward cache used by backward propagation."""
        self._engine.clear_cache()
