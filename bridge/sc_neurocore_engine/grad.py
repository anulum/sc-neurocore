# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Surrogate gradient wrappers for the v3 Rust engine

"""Surrogate gradient wrappers for the v3 Rust engine."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from sc_neurocore_engine.sc_neurocore_engine import (
    DifferentiableDenseLayer as _RustDifferentiableDenseLayer,
)
from sc_neurocore_engine.sc_neurocore_engine import SurrogateLif as _RustSurrogateLif


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
        self._engine = _RustSurrogateLif(
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
        return self._engine.forward(leak_k, gain_k, i_t, noise_in)

    def backward(self, grad_output: float) -> float:
        return float(self._engine.backward(float(grad_output)))

    def clear_trace(self) -> None:
        self._engine.clear_trace()

    def reset(self) -> None:
        self._engine.reset()

    def trace_len(self) -> int:
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
        return np.asarray(self._engine.get_weights(), dtype=np.float64)

    def forward(self, input_values: Sequence[float], seed: int = 44257) -> npt.NDArray[np.float64]:
        values = np.asarray(input_values, dtype=np.float64)
        out = self._engine.forward(values.tolist(), int(seed))
        return np.asarray(out, dtype=np.float64)

    def backward(
        self, grad_output: Sequence[float]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        grad = np.asarray(grad_output, dtype=np.float64)
        grad_in, grad_w = self._engine.backward(grad.tolist())
        return np.asarray(grad_in, dtype=np.float64), np.asarray(grad_w, dtype=np.float64)

    def update_weights(self, weight_grads: Sequence[Sequence[float]], lr: float) -> None:
        grads = np.asarray(weight_grads, dtype=np.float64)
        self._engine.update_weights(grads.tolist(), float(lr))

    def clear_cache(self) -> None:
        self._engine.clear_cache()
