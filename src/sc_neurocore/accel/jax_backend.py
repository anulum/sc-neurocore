# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JAX backend for SC-NeuroCore.

Provides JAX-accelerated primitives for stochastic computing, unlocking
automatic differentiation, JIT compilation (XLA), and native TPU/GPU scaling.

Usage::

    from sc_neurocore.accel.jax_backend import jnp, HAS_JAX, to_jax, to_host
    from sc_neurocore.accel.jax_backend import jax_pack_bitstream, jax_vec_mac

    if HAS_JAX:
        bits = jnp.array([1, 0, 1, 1], dtype=jnp.uint8)
        packed = jax_pack_bitstream(bits)
"""

from typing import Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

__all__ = [
    "jax",
    "jnp",
    "HAS_JAX",
    "to_jax",
    "to_host",
    "jax_pack_bitstream",
    "jax_vec_and",
    "jax_popcount",
    "jax_vec_mac",
    "jax_lif_step",
    "jax_forward_pass",
    "jax_surrogate_gradient_step",
]


def to_jax(arr: Any) -> Any:
    """Move a NumPy array to the JAX device."""
    if HAS_JAX:
        return jnp.asarray(arr)
    return arr


def to_host(arr: Any) -> np.ndarray[Any, Any]:
    """Bring a JAX array back to host RAM as a NumPy array."""
    if HAS_JAX and isinstance(arr, jax.Array):
        return np.asarray(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# JAX-accelerated bitstream primitives
# ---------------------------------------------------------------------------

if HAS_JAX:

    @jax.jit
    def _jax_pack_1d(bits: jax.Array) -> jax.Array:
        length = bits.size
        pad = (64 - length % 64) % 64
        if pad:
            bits = jnp.concatenate([bits, jnp.zeros(pad, dtype=jnp.uint8)])
        chunks = bits.reshape(-1, 64)
        powers = jnp.uint64(1) << jnp.arange(64, dtype=jnp.uint64)
        return jnp.sum(chunks.astype(jnp.uint64) * powers, axis=1)

    @jax.jit
    def _jax_pack_2d(bits: jax.Array) -> jax.Array:
        B, length = bits.shape
        pad = (64 - length % 64) % 64
        if pad:
            bits = jnp.concatenate([bits, jnp.zeros((B, pad), dtype=jnp.uint8)], axis=1)
        n_words = bits.shape[1] // 64
        chunks = bits.reshape(B, n_words, 64)
        powers = jnp.uint64(1) << jnp.arange(64, dtype=jnp.uint64)
        return jnp.sum(chunks.astype(jnp.uint64) * powers, axis=2)


def jax_pack_bitstream(bits: Any) -> Any:
    """
    Pack uint8 {0,1} array into uint64 words using JAX.
    """
    if not HAS_JAX:
        from sc_neurocore.exceptions import SCDependencyError

        raise SCDependencyError("JAX is not available.")

    bits = jnp.asarray(bits, dtype=jnp.uint8)

    if bits.ndim == 1:
        return _jax_pack_1d(bits)
    elif bits.ndim == 2:
        return _jax_pack_2d(bits)

    from sc_neurocore.exceptions import SCEncodingError

    raise SCEncodingError(f"Expected 1-D or 2-D, got {bits.ndim}-D")


if HAS_JAX:

    @jax.jit
    def jax_vec_and(a: jax.Array, b: jax.Array) -> jax.Array:
        """Bitwise AND on packed uint64 arrays (SC multiplication)."""
        return jnp.bitwise_and(a, b)

    @jax.jit
    def jax_popcount(packed: jax.Array) -> jax.Array:
        """
        Vectorised SWAR popcount on uint64 arrays using JAX.
        """
        x = packed.astype(jnp.uint64)
        m1 = jnp.uint64(0x5555555555555555)
        m2 = jnp.uint64(0x3333333333333333)
        m4 = jnp.uint64(0x0F0F0F0F0F0F0F0F)
        h01 = jnp.uint64(0x0101010101010101)

        x = x - ((x >> jnp.uint64(1)) & m1)
        x = (x & m2) + ((x >> jnp.uint64(2)) & m2)
        x = (x + (x >> jnp.uint64(4))) & m4
        res: jax.Array = (x * h01) >> jnp.uint64(56)
        return res

    @jax.jit
    def jax_vec_mac(packed_weights: jax.Array, packed_inputs: jax.Array) -> jax.Array:
        """
        JAX-accelerated multiply-accumulate for a dense SC layer.
        """
        products = jnp.bitwise_and(packed_weights, packed_inputs[None, :, :])
        counts = jax_popcount(products)
        res: jax.Array = jnp.sum(counts, axis=(1, 2))
        return res

    @jax.jit
    def jax_lif_step(
        v: jax.Array,
        I_t: jax.Array,
        v_rest: float,
        v_reset: float,
        v_threshold: float,
        alpha: float,
        resistance: float,
        noise: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Vectorized LIF step using JAX.

        dv = (v_rest - v) * alpha + I_t * resistance + noise
        """
        v_next = v + (v_rest - v) * alpha + I_t * resistance + noise
        spikes = v_next >= v_threshold
        v_next = jnp.where(spikes, v_reset, v_next)
        return v_next, spikes.astype(jnp.uint8)

    def jax_forward_pass(
        weights: list[jax.Array],
        x: jax.Array,
        n_steps: int,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_threshold: float = 1.0,
        alpha: float = 0.9,
    ) -> tuple[list[jax.Array], jax.Array]:
        """
        Multi-layer SNN forward pass with LIF neurons.

        Returns (spike_trains_per_layer, final_membrane_potentials).
        Each layer: s = Heaviside(v - threshold), v = alpha * v * (1-s) + W @ s_prev
        """
        batch = x.shape[0]
        spikes = x
        all_spikes = []

        for W in weights:
            n_out = W.shape[0]
            v = jnp.full((batch, n_out), v_rest)
            layer_spikes = []

            for _t in range(n_steps):
                current = spikes @ W.T
                v = alpha * v * (1.0 - v_reset) + current
                s = (v >= v_threshold).astype(jnp.float32)
                v = jnp.where(s > 0.5, v_reset, v)
                layer_spikes.append(s)

            # Output spikes = mean firing rate over time
            spikes = jnp.stack(layer_spikes, axis=0).mean(axis=0)
            all_spikes.append(jnp.stack(layer_spikes, axis=0))

        return all_spikes, v

    def jax_surrogate_gradient_step(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int = 25,
        lr: float = 1e-3,
        beta: float = 10.0,
    ) -> tuple[list[jax.Array], float]:
        """
        One training step with surrogate gradient (fast sigmoid).

        Uses jax.grad on a cross-entropy loss over mean output spike rates.
        Returns (updated_weights, loss_value).
        """

        def loss_fn(ws):
            batch = x.shape[0]
            spikes_in = x
            for W in ws:
                n_out = W.shape[0]
                v = jnp.zeros((batch, n_out))
                spike_sum = jnp.zeros((batch, n_out))
                for _t in range(n_steps):
                    current = spikes_in @ W.T
                    v = 0.9 * v + current
                    # Fast sigmoid surrogate: σ(β(v-θ)) / β
                    sg = 1.0 / (1.0 + jnp.abs(beta * (v - 1.0)))
                    spike_sum = spike_sum + sg
                    v = v * (1.0 - (v >= 1.0).astype(v.dtype))
                spikes_in = spike_sum / n_steps
            logits = spikes_in
            log_softmax = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
            ce = -jnp.sum(targets * log_softmax) / batch
            return ce

        loss_val, grads = jax.value_and_grad(loss_fn)(weights)
        updated = [w - lr * g for w, g in zip(weights, grads)]
        return updated, float(loss_val)
