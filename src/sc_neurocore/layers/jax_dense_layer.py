# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
from typing import Dict, Any, Optional

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
    neuron_params: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not HAS_JAX:
            from sc_neurocore.exceptions import SCDependencyError

            raise SCDependencyError("JAX is required for JaxSCDenseLayer.")

        if self.neuron_params is None:
            self.neuron_params = {}

        # Layer Parameters (JAX Arrays)
        self.v_rest = float(self.neuron_params.get("v_rest", LIF_V_REST))
        self.v_reset = float(self.neuron_params.get("v_reset", LIF_V_RESET))
        self.v_threshold = float(self.neuron_params.get("v_threshold", LIF_V_THRESHOLD))
        self.tau_mem = float(self.neuron_params.get("tau_mem", LIF_TAU_MEM))
        self.resistance = float(self.neuron_params.get("resistance", LIF_RESISTANCE))
        self.noise_std = float(self.neuron_params.get("noise_std", LIF_LAYER_NOISE_STD))
        self.alpha = float(self.dt_ms / self.tau_mem)

        # State (JAX Arrays)
        self.v = jnp.full((self.n_neurons,), self.v_rest)

        # RNG State
        self.rng_key = jax.random.PRNGKey(self.seed or 42)

    def step(self, I_t: jax.Array) -> jax.Array:
        """
        Advance the entire layer by one time step.

        I_t: (n_neurons,) input current for each neuron.
        Returns:
        spikes: (n_neurons,) uint8 array.
        """
        # Generate noise
        self.rng_key, subkey = jax.random.split(self.rng_key)
        noise = jax.random.normal(subkey, (self.n_neurons,)) * self.noise_std

        # Update neurons
        self.v, spikes = jax_lif_step(
            self.v,
            I_t,
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

        T = currents.shape[0]
        all_spikes = []

        for t in range(T):
            all_spikes.append(self.step(currents[t]))

        return jnp.stack(all_spikes)

    def reset(self) -> None:
        self.v = jnp.full((self.n_neurons,), self.v_rest)
