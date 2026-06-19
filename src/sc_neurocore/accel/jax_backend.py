# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX backend for SC-NeuroCore

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

from __future__ import annotations

import math
import types
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp: types.ModuleType = np  # type: ignore[no-redef]
    jax = None  # type: ignore[assignment]

__all__ = [
    "jax",
    "jnp",
    "HAS_JAX",
    "JAX_SURROGATE_PATHS",
    "to_jax",
    "to_host",
    "jax_pack_bitstream",
    "jax_vec_and",
    "jax_popcount",
    "jax_vec_mac",
    "jax_lif_step",
    "jax_forward_pass",
    "jax_surrogate_loss",
    "jax_surrogate_gradient_step",
]

JAX_SURROGATE_PATHS = ("custom_vjp", "legacy_stop_gradient")


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

    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def _validate_positive_finite_scalar(name: str, value: float) -> None:
        if not isinstance(value, int | float) or not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite scalar")

    def _validate_finite_scalar(name: str, value: float) -> None:
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    def _validate_floating_array(name: str, value: Any) -> jax.Array:
        arr = jnp.asarray(value)
        if not np.issubdtype(np.dtype(arr.dtype), np.floating):
            raise ValueError(f"{name} must be a floating-point array")
        return arr

    def _validate_lif_array(
        name: str, value: Any, expected_shape: tuple[int, ...] | None = None
    ) -> jax.Array:
        arr = _validate_floating_array(name, value)
        if arr.size == 0 or 0 in arr.shape:
            raise ValueError(f"{name} must be a non-empty floating-point array")
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} must match v shape {expected_shape}")
        _validate_concrete_finite_array(name, arr)
        return arr

    def _validate_uint64_array(name: str, value: Any) -> jax.Array:
        arr = jnp.asarray(value)
        if arr.dtype != jnp.uint64:
            raise ValueError(f"{name} must be a non-empty uint64 array")
        if arr.size == 0 or 0 in arr.shape:
            raise ValueError(f"{name} must be a non-empty uint64 array")
        return arr

    def _validate_concrete_finite_array(name: str, value: Any) -> None:
        if isinstance(value, jax.core.Tracer):
            return
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")

    def _validate_surrogate_inputs(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int,
        beta: float,
        threshold: float,
        surrogate_path: str,
    ) -> tuple[list[jax.Array], jax.Array, jax.Array]:
        _validate_positive_int("n_steps", n_steps)
        _validate_positive_finite_scalar("beta", beta)
        _validate_finite_scalar("threshold", threshold)

        if surrogate_path not in JAX_SURROGATE_PATHS:
            valid = ", ".join(JAX_SURROGATE_PATHS)
            raise ValueError(f"Unknown surrogate_path {surrogate_path!r}; expected one of: {valid}")

        if len(weights) == 0:
            raise ValueError("weights must contain at least one layer")

        x_arr = _validate_floating_array("x", x)
        targets_arr = _validate_floating_array("targets", targets)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2-D, got {x_arr.ndim}-D")
        if targets_arr.ndim != 2:
            raise ValueError(f"targets must be 2-D, got {targets_arr.ndim}-D")
        if x_arr.shape[0] != targets_arr.shape[0]:
            raise ValueError(
                f"targets batch dimension {targets_arr.shape[0]} does not match x batch "
                f"dimension {x_arr.shape[0]}"
            )
        _validate_concrete_finite_array("x", x_arr)
        _validate_concrete_finite_array("targets", targets_arr)

        validated_weights = []
        expected_input_dim = x_arr.shape[1]
        for idx, weight in enumerate(weights):
            weight_arr = _validate_floating_array(f"weights[{idx}]", weight)
            if weight_arr.ndim != 2:
                raise ValueError(f"weights[{idx}] must be 2-D, got {weight_arr.ndim}-D")
            if 0 in weight_arr.shape:
                raise ValueError(f"weights[{idx}] must not have empty dimensions")
            if weight_arr.shape[1] != expected_input_dim:
                raise ValueError(
                    f"weights[{idx}] input dimension {weight_arr.shape[1]} does not match "
                    f"previous layer dimension {expected_input_dim}"
                )
            _validate_concrete_finite_array(f"weights[{idx}]", weight_arr)
            validated_weights.append(weight_arr)
            expected_input_dim = weight_arr.shape[0]

        if targets_arr.shape[1] != expected_input_dim:
            raise ValueError(
                f"targets output dimension {targets_arr.shape[1]} does not match final "
                f"layer dimension {expected_input_dim}"
            )

        return validated_weights, x_arr, targets_arr

    def _validate_forward_inputs(
        weights: list[jax.Array],
        x: Any,
        n_steps: int,
        v_rest: float,
        v_reset: float,
        v_threshold: float,
        alpha: float,
    ) -> tuple[list[jax.Array], jax.Array]:
        _validate_positive_int("n_steps", n_steps)
        _validate_finite_scalar("v_rest", v_rest)
        _validate_finite_scalar("v_reset", v_reset)
        _validate_finite_scalar("v_threshold", v_threshold)
        _validate_positive_finite_scalar("alpha", alpha)
        if len(weights) == 0:
            raise ValueError("weights must contain at least one layer")

        x_arr = _validate_floating_array("x", x)
        if x_arr.ndim != 2:
            raise ValueError(f"x must be 2-D, got {x_arr.ndim}-D")
        if 0 in x_arr.shape:
            raise ValueError("x must not have empty dimensions")
        _validate_concrete_finite_array("x", x_arr)

        validated_weights = []
        expected_input_dim = x_arr.shape[1]
        for idx, weight in enumerate(weights):
            weight_arr = _validate_floating_array(f"weights[{idx}]", weight)
            if weight_arr.ndim != 2:
                raise ValueError(f"weights[{idx}] must be 2-D, got {weight_arr.ndim}-D")
            if 0 in weight_arr.shape:
                raise ValueError(f"weights[{idx}] must not have empty dimensions")
            if weight_arr.shape[1] != expected_input_dim:
                raise ValueError(
                    f"weights[{idx}] input dimension {weight_arr.shape[1]} does not match "
                    f"previous layer dimension {expected_input_dim}"
                )
            _validate_concrete_finite_array(f"weights[{idx}]", weight_arr)
            validated_weights.append(weight_arr)
            expected_input_dim = weight_arr.shape[0]

        return validated_weights, x_arr

    def _fast_sigmoid_proxy(v: jax.Array, beta: jax.Array, threshold: jax.Array) -> jax.Array:
        centered = v - threshold
        return centered / (1.0 + jnp.abs(beta * centered))

    def _fast_sigmoid_grad(v: jax.Array, beta: jax.Array, threshold: jax.Array) -> jax.Array:
        centered = v - threshold
        return 1.0 / (1.0 + jnp.abs(beta * centered)) ** 2

    @jax.custom_vjp
    def _custom_vjp_superspike(v: jax.Array, beta: jax.Array, threshold: jax.Array) -> jax.Array:
        return (v >= threshold).astype(v.dtype)

    def _custom_vjp_superspike_fwd(
        v: jax.Array, beta: jax.Array, threshold: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        return _custom_vjp_superspike(v, beta, threshold), (v, beta, threshold)

    def _custom_vjp_superspike_bwd(
        res: tuple[jax.Array, jax.Array, jax.Array], g: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        v, beta, threshold = res
        grad = _fast_sigmoid_grad(v, beta, threshold)
        return g * grad, jnp.zeros_like(beta), jnp.zeros_like(threshold)

    _custom_vjp_superspike.defvjp(
        _custom_vjp_superspike_fwd,
        _custom_vjp_superspike_bwd,
    )

    def _legacy_stop_gradient_spike(
        v: jax.Array, beta: jax.Array, threshold: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        centered = v - threshold
        surrogate_rate = 1.0 / (1.0 + jnp.abs(beta * centered))
        spike_hard = (v >= threshold).astype(v.dtype)
        spike_reset = surrogate_rate + jax.lax.stop_gradient(spike_hard - surrogate_rate)
        return surrogate_rate, spike_reset

    def _jax_loss_with_custom_vjp_surrogate(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int,
        beta: float,
        threshold: float,
    ) -> jax.Array:
        batch = x.shape[0]
        spikes_in = x
        beta_arr = jnp.asarray(beta, dtype=x.dtype)
        threshold_arr = jnp.asarray(threshold, dtype=x.dtype)

        for W in weights:
            n_out = W.shape[0]
            v = jnp.zeros((batch, n_out), dtype=x.dtype)
            spike_sum = jnp.zeros((batch, n_out), dtype=x.dtype)
            for _t in range(n_steps):
                current = spikes_in @ W.T
                v = 0.9 * v + current
                spikes = _custom_vjp_superspike(v, beta_arr, threshold_arr)
                spike_sum = spike_sum + spikes
                v = v * (1.0 - spikes)
            spikes_in = spike_sum / n_steps

        logits = spikes_in
        log_softmax = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        ce = -jnp.sum(targets * log_softmax) / batch
        return ce

    def _jax_loss_with_legacy_stop_gradient_surrogate(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int,
        beta: float,
        threshold: float,
    ) -> jax.Array:
        batch = x.shape[0]
        spikes_in = x
        beta_arr = jnp.asarray(beta, dtype=x.dtype)
        threshold_arr = jnp.asarray(threshold, dtype=x.dtype)

        for W in weights:
            n_out = W.shape[0]
            v = jnp.zeros((batch, n_out), dtype=x.dtype)
            spike_sum = jnp.zeros((batch, n_out), dtype=x.dtype)
            for _t in range(n_steps):
                current = spikes_in @ W.T
                v = 0.9 * v + current
                surrogate_rate, spike_reset = _legacy_stop_gradient_spike(
                    v, beta_arr, threshold_arr
                )
                spike_sum = spike_sum + surrogate_rate
                v = v * (1.0 - spike_reset)
            spikes_in = spike_sum / n_steps

        logits = spikes_in
        log_softmax = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        ce = -jnp.sum(targets * log_softmax) / batch
        return ce

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

    from sc_neurocore.exceptions import SCEncodingError

    host_bits = np.asarray(bits)
    if host_bits.dtype != np.uint8:
        raise SCEncodingError("Expected a uint8 binary bitstream containing only 0 and 1")
    if host_bits.ndim not in (1, 2):
        raise SCEncodingError(f"Expected 1-D or 2-D, got {host_bits.ndim}-D")
    if host_bits.size == 0 or 0 in host_bits.shape:
        raise SCEncodingError("Expected a non-empty uint8 binary bitstream")
    if not np.isin(host_bits, np.array([0, 1], dtype=np.uint8)).all():
        raise SCEncodingError("Expected a uint8 binary bitstream containing only 0 and 1")

    bits = jnp.asarray(host_bits, dtype=jnp.uint8)

    if bits.ndim == 1:
        return _jax_pack_1d(bits)
    if bits.ndim == 2:
        return _jax_pack_2d(bits)

    raise SCEncodingError(f"Expected 1-D or 2-D, got {bits.ndim}-D")


if HAS_JAX:

    @jax.jit
    def _jax_vec_and_impl(a: jax.Array, b: jax.Array) -> jax.Array:
        """Bitwise AND on packed uint64 arrays (SC multiplication)."""
        return jnp.bitwise_and(a, b)

    def jax_vec_and(a: Any, b: Any) -> jax.Array:
        """Bitwise AND on matching non-empty uint64 packed arrays."""
        a_arr = _validate_uint64_array("a", a)
        b_arr = _validate_uint64_array("b", b)
        if a_arr.shape != b_arr.shape:
            raise ValueError(f"a shape {a_arr.shape} must match b shape {b_arr.shape}")
        result: jax.Array = _jax_vec_and_impl(a_arr, b_arr)
        return result

    @jax.jit
    def _jax_popcount_impl(packed: jax.Array) -> jax.Array:
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

    def jax_popcount(packed: Any) -> jax.Array:
        """
        Vectorised SWAR popcount on a non-empty uint64 array using JAX.
        """
        packed_arr = _validate_uint64_array("packed", packed)
        result: jax.Array = _jax_popcount_impl(packed_arr)
        return result

    @jax.jit
    def _jax_vec_mac_impl(packed_weights: jax.Array, packed_inputs: jax.Array) -> jax.Array:
        """
        JAX-accelerated multiply-accumulate for a dense SC layer.
        """
        products = jnp.bitwise_and(packed_weights, packed_inputs[None, :, :])
        counts = _jax_popcount_impl(products)
        res: jax.Array = jnp.sum(counts, axis=(1, 2))
        return res

    def jax_vec_mac(packed_weights: Any, packed_inputs: Any) -> jax.Array:
        """
        JAX-accelerated multiply-accumulate for packed uint64 dense SC layers.
        """
        weight_arr = _validate_uint64_array("packed_weights", packed_weights)
        input_arr = _validate_uint64_array("packed_inputs", packed_inputs)
        if weight_arr.ndim != 3:
            raise ValueError(f"packed_weights must be 3-D, got {weight_arr.ndim}-D")
        if input_arr.ndim != 2:
            raise ValueError(f"packed_inputs must be 2-D, got {input_arr.ndim}-D")
        if weight_arr.shape[1] != input_arr.shape[0]:
            raise ValueError(
                f"packed_weights input dimension {weight_arr.shape[1]} does not match "
                f"packed_inputs input dimension {input_arr.shape[0]}"
            )
        if weight_arr.shape[2] != input_arr.shape[1]:
            raise ValueError(
                f"packed_weights word dimension {weight_arr.shape[2]} does not match "
                f"packed_inputs word dimension {input_arr.shape[1]}"
            )
        result: jax.Array = _jax_vec_mac_impl(weight_arr, input_arr)
        return result

    @jax.jit
    def _jax_lif_step_impl(
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

    def jax_lif_step(
        v: Any,
        I_t: Any,
        v_rest: float,
        v_reset: float,
        v_threshold: float,
        alpha: float,
        resistance: float,
        noise: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Vectorized LIF step using JAX with fail-closed public input guards.

        dv = (v_rest - v) * alpha + I_t * resistance + noise
        """
        _validate_finite_scalar("v_rest", v_rest)
        _validate_finite_scalar("v_reset", v_reset)
        _validate_finite_scalar("v_threshold", v_threshold)
        _validate_positive_finite_scalar("alpha", alpha)
        _validate_positive_finite_scalar("resistance", resistance)
        v_arr = _validate_lif_array("v", v)
        current_arr = _validate_lif_array("I_t", I_t, expected_shape=v_arr.shape)
        noise_arr = _validate_lif_array("noise", noise, expected_shape=v_arr.shape)
        result: tuple[jax.Array, jax.Array] = _jax_lif_step_impl(
            v_arr,
            current_arr,
            v_rest,
            v_reset,
            v_threshold,
            alpha,
            resistance,
            noise_arr,
        )
        return result

    def jax_forward_pass(
        weights: list[jax.Array],
        x: Any,
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
        weights, x = _validate_forward_inputs(
            weights=weights,
            x=x,
            n_steps=n_steps,
            v_rest=v_rest,
            v_reset=v_reset,
            v_threshold=v_threshold,
            alpha=alpha,
        )
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

    def jax_surrogate_loss(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int = 25,
        beta: float = 10.0,
        threshold: float = 1.0,
        surrogate_path: str = "custom_vjp",
    ) -> jax.Array:
        """
        Cross-entropy loss for JAX SNN training with explicit surrogate paths.

        Available paths:
        - ``custom_vjp``: hard spikes forward, fast-sigmoid proxy backward
          via ``jax.custom_vjp``
        - ``legacy_stop_gradient``: historical straight-through reset path
          using ``jax.lax.stop_gradient``
        """

        weights, x, targets = _validate_surrogate_inputs(
            weights=weights,
            x=x,
            targets=targets,
            n_steps=n_steps,
            beta=beta,
            threshold=threshold,
            surrogate_path=surrogate_path,
        )

        if surrogate_path == "custom_vjp":
            return _jax_loss_with_custom_vjp_surrogate(
                weights=weights,
                x=x,
                targets=targets,
                n_steps=n_steps,
                beta=beta,
                threshold=threshold,
            )
        return _jax_loss_with_legacy_stop_gradient_surrogate(
            weights=weights,
            x=x,
            targets=targets,
            n_steps=n_steps,
            beta=beta,
            threshold=threshold,
        )

    def jax_surrogate_gradient_step(
        weights: list[jax.Array],
        x: jax.Array,
        targets: jax.Array,
        n_steps: int = 25,
        lr: float = 1e-3,
        beta: float = 10.0,
        threshold: float = 1.0,
        surrogate_path: str = "custom_vjp",
    ) -> tuple[list[jax.Array], float]:
        """
        One training step with surrogate gradient over an explicit JAX path.

        ``custom_vjp`` is the modern path. ``legacy_stop_gradient`` keeps the
        historical training route available for side-by-side verification.
        """

        _validate_positive_finite_scalar("lr", lr)
        weights, x, targets = _validate_surrogate_inputs(
            weights=weights,
            x=x,
            targets=targets,
            n_steps=n_steps,
            beta=beta,
            threshold=threshold,
            surrogate_path=surrogate_path,
        )

        def loss_fn(ws: list[jax.Array]) -> jax.Array:
            return jax_surrogate_loss(
                weights=ws,
                x=x,
                targets=targets,
                n_steps=n_steps,
                beta=beta,
                threshold=threshold,
                surrogate_path=surrogate_path,
            )

        loss_val, grads = jax.value_and_grad(loss_fn)(weights)
        updated = [w - lr * g for w, g in zip(weights, grads)]
        return updated, float(loss_val)


else:
    # Fallbacks when JAX is not installed — raise clear error on use.

    def _jax_not_installed(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("JAX is not installed. Install with: pip install sc-neurocore[jax]")

    jax_vec_and = _jax_not_installed
    jax_popcount = _jax_not_installed
    jax_vec_mac = _jax_not_installed
    jax_lif_step = _jax_not_installed
    jax_forward_pass = _jax_not_installed
    jax_surrogate_loss = _jax_not_installed
    jax_surrogate_gradient_step = _jax_not_installed
