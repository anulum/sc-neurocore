# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX-accelerated Stochastic Dense Layer

"""
JAX-accelerated Stochastic Dense Layer.

This module implements a dense layer of LIF neurons using JAX for
vectorized, JIT-compiled execution. It provides a massive performance
boost over the loop-based SCDenseLayer and enables native TPU/GPU scaling.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np

from sc_neurocore.accel.jax_backend import jnp, jax, HAS_JAX, jax_lif_step
from sc_neurocore.constants import (
    LAYER_DEFAULT_LENGTH,
    LIF_DT,
    LIF_V_REST,
    LIF_V_RESET,
    LIF_V_THRESHOLD,
    LIF_TAU_MEM,
    LIF_RESISTANCE,
    LIF_LAYER_NOISE_STD,
)

_ALLOWED_NEURON_PARAMS = frozenset(
    {
        "v_rest",
        "v_reset",
        "v_threshold",
        "tau_mem",
        "resistance",
        "noise_std",
    }
)
_MAX_JAX_SEED = 2**32 - 1


@dataclass
class JaxSCDenseLayer:
    """
    JAX-accelerated stochastic dense layer of LIF neurons.

    Example
    -------
    >>> layer = JaxSCDenseLayer(n_neurons=10, n_inputs=5, seed=0)  # doctest: +SKIP
    >>> import jax.numpy as jnp  # doctest: +SKIP
    >>> spikes = layer.step(jnp.ones(10) * 0.5)  # doctest: +SKIP
    >>> spikes.shape  # doctest: +SKIP
    (10,)
    """

    n_neurons: int
    n_inputs: int
    bitstream_length: int = LAYER_DEFAULT_LENGTH
    dt_ms: float = LIF_DT
    neuron_params: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    weights: Any = None

    def __post_init__(self) -> None:
        if not HAS_JAX:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("JAX is required for JaxSCDenseLayer.")

        if self.neuron_params is None:
            self.neuron_params = {}

        self._validate_config()

        self.v_rest = self._finite_param("v_rest", self.neuron_params.get("v_rest", LIF_V_REST))
        self.v_reset = self._finite_param("v_reset", self.neuron_params.get("v_reset", LIF_V_RESET))
        self.v_threshold = self._finite_param(
            "v_threshold", self.neuron_params.get("v_threshold", LIF_V_THRESHOLD)
        )
        self.tau_mem = self._positive_param(
            "tau_mem", self.neuron_params.get("tau_mem", LIF_TAU_MEM)
        )
        self.resistance = self._positive_param(
            "resistance", self.neuron_params.get("resistance", LIF_RESISTANCE)
        )
        self.noise_std = self._nonnegative_param(
            "noise_std", self.neuron_params.get("noise_std", LIF_LAYER_NOISE_STD)
        )
        self.alpha = float(self.dt_ms / self.tau_mem)

        self.v = jnp.full((self.n_neurons,), self.v_rest)
        self.rng_key = jax.random.PRNGKey(self.seed if self.seed is not None else 42)
        self.weights = self._initialise_weights()

    @staticmethod
    def _positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _finite_param(name: str, value: Any) -> float:
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    @classmethod
    def _positive_param(cls, name: str, value: Any) -> float:
        result = cls._finite_param(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @classmethod
    def _nonnegative_param(cls, name: str, value: Any) -> float:
        result = cls._finite_param(name, value)
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return result

    def _validate_config(self) -> None:
        self._positive_int("n_neurons", self.n_neurons)
        self._positive_int("n_inputs", self.n_inputs)
        self._positive_int("bitstream_length", self.bitstream_length)
        self.dt_ms = self._positive_param("dt_ms", self.dt_ms)
        unknown = set(self.neuron_params or {}) - _ALLOWED_NEURON_PARAMS
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"neuron_params contains unknown keys: {joined}")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError("seed must be a non-negative integer")
            if self.seed < 0 or self.seed > _MAX_JAX_SEED:
                raise ValueError(f"seed must be in [0, {_MAX_JAX_SEED}]")

    def _initialise_weights(self) -> Any:
        if self.weights is None:
            weight_key = jax.random.PRNGKey(self.seed if self.seed is not None else 42)
            return jax.random.normal(weight_key, (self.n_neurons, self.n_inputs)) * 0.1

        arr = jnp.asarray(self.weights)
        if not np.issubdtype(np.dtype(arr.dtype), np.floating):
            raise ValueError("weights must be a floating-point array")
        if arr.shape != (self.n_neurons, self.n_inputs):
            raise ValueError(
                f"weights shape {arr.shape} must match layer shape {(self.n_neurons, self.n_inputs)}"
            )
        if not np.isfinite(np.asarray(arr)).all():
            raise ValueError("weights must contain only finite values")
        return arr

    def _validate_current_vector(self, currents: Any) -> Any:
        arr = jnp.asarray(currents)
        if not np.issubdtype(np.dtype(arr.dtype), np.floating):
            raise ValueError("I_t must be a floating-point array")
        if arr.shape not in ((self.n_neurons,), (self.n_inputs,)):
            raise ValueError(
                f"I_t shape {arr.shape} must match layer current shape {(self.n_neurons,)} "
                f"or input shape {(self.n_inputs,)}"
            )
        if not np.isfinite(np.asarray(arr)).all():
            raise ValueError("I_t must contain only finite values")
        return arr

    def _current_from_step_input(self, values: Any) -> Any:
        arr = self._validate_current_vector(values)
        if arr.shape == (self.n_inputs,) and self.n_inputs != self.n_neurons:
            return self.weights @ arr
        return arr

    def _validate_current_sequence(self, currents: Any) -> Any:
        arr = jnp.asarray(currents)
        if not np.issubdtype(np.dtype(arr.dtype), np.floating):
            raise ValueError("currents must be a floating-point array")
        if arr.ndim != 2:
            raise ValueError(f"currents must be 2-D, got {arr.ndim}-D")
        if arr.shape[0] == 0:
            raise ValueError("currents must be non-empty")
        if arr.shape[1] not in (self.n_neurons, self.n_inputs):
            raise ValueError(
                f"currents shape {arr.shape} must match time x layer shape (*, {self.n_neurons}) "
                f"or time x input shape (*, {self.n_inputs})"
            )
        if not np.isfinite(np.asarray(arr)).all():
            raise ValueError("currents must contain only finite values")
        return arr

    def step(self, I_t: jax.Array) -> jax.Array:
        """
        Advance the entire layer by one time step.

        I_t: (n_neurons,) input current for each neuron.
        Returns:
        spikes: (n_neurons,) uint8 array.
        """
        current = self._current_from_step_input(I_t)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.normal(subkey, (self.n_neurons,)) * self.noise_std

        self.v, spikes = jax_lif_step(
            self.v,
            current,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.alpha,
            self.resistance,
            noise,
        )
        res: jax.Array = spikes
        return res

    def run(self, currents: jax.Array) -> jax.Array:
        """
        Run for multiple steps.

        currents: (T, n_neurons)
        Returns:
        spikes: (T, n_neurons)
        """
        # Note: In a production JAX implementation, we would use jax.lax.scan
        # for maximum performance.

        currents = self._validate_current_sequence(currents)
        T = currents.shape[0]
        all_spikes = []

        for t in range(T):
            all_spikes.append(self.step(currents[t]))

        return jnp.stack(all_spikes)

    def reset(self) -> None:
        self.v = jnp.full((self.n_neurons,), self.v_rest)
