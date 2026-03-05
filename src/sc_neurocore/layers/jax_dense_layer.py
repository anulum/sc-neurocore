# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JAX-accelerated Stochastic Dense Layer.

This module implements a dense layer of LIF neurons using JAX for
vectorized, JIT-compiled execution. It provides a massive performance
boost over the loop-based SCDenseLayer and enables native TPU/GPU scaling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

from sc_neurocore.accel.jax_backend import jnp, jax, HAS_JAX, to_jax, to_host, jax_lif_step
from sc_neurocore.utils.rng import RNG


@dataclass
class JaxSCDenseLayer:
    """
    Stochastic Dense Layer implemented in JAX.
    """

    n_neurons: int
    n_inputs: int
    bitstream_length: int = 1024
    dt_ms: float = 1.0
    neuron_params: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not HAS_JAX:
            raise RuntimeError("JAX is required for JaxSCDenseLayer.")

        if self.neuron_params is None:
            self.neuron_params = {}

        # Layer Parameters (JAX Arrays)
        self.v_rest = float(self.neuron_params.get("v_rest", 0.0))
        self.v_reset = float(self.neuron_params.get("v_reset", 0.0))
        self.v_threshold = float(self.neuron_params.get("v_threshold", 1.0))
        self.tau_mem = float(self.neuron_params.get("tau_mem", 20.0))
        self.resistance = float(self.neuron_params.get("resistance", 1.0))
        self.noise_std = float(self.neuron_params.get("noise_std", 0.02))
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
