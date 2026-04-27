# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for explicit JAX surrogate execution paths

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from sc_neurocore.accel.jax_backend import (
    JAX_SURROGATE_PATHS,
    _custom_vjp_superspike,
    jax_surrogate_gradient_step,
    jax_surrogate_loss,
)


def _numpy_fast_sigmoid_proxy(values: np.ndarray, beta: float, threshold: float) -> np.ndarray:
    centered = values - threshold
    return centered / (1.0 + np.abs(beta * centered))


def test_jax_surrogate_paths_are_explicit():
    assert JAX_SURROGATE_PATHS == ("custom_vjp", "legacy_stop_gradient")


def test_custom_vjp_gradient_matches_numpy_proxy_finite_difference():
    beta = 6.0
    threshold = 1.0
    voltages = np.array([0.65, 1.35], dtype=np.float64)

    def grad_target(values: jax.Array) -> jax.Array:
        beta_arr = jnp.asarray(beta, dtype=values.dtype)
        threshold_arr = jnp.asarray(threshold, dtype=values.dtype)
        return jnp.sum(_custom_vjp_superspike(values, beta_arr, threshold_arr))

    autodiff = np.asarray(jax.grad(grad_target)(jnp.asarray(voltages)))

    eps = 1e-6
    finite_diff = np.empty_like(voltages)
    for idx in range(voltages.size):
        plus = voltages.copy()
        minus = voltages.copy()
        plus[idx] += eps
        minus[idx] -= eps
        forward = _numpy_fast_sigmoid_proxy(plus, beta, threshold).sum()
        backward = _numpy_fast_sigmoid_proxy(minus, beta, threshold).sum()
        finite_diff[idx] = (forward - backward) / (2.0 * eps)

    assert np.allclose(autodiff, finite_diff, rtol=1e-3, atol=1e-3)


def test_jax_surrogate_loss_rejects_unknown_path():
    x = jnp.asarray([[1.2, 0.0]], dtype=jnp.float32)
    targets = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    weights = [jnp.asarray([[0.5, 0.1], [0.0, 0.4]], dtype=jnp.float32)]

    with pytest.raises(ValueError, match="surrogate_path"):
        jax_surrogate_loss(weights, x, targets, surrogate_path="unknown")


def test_custom_vjp_path_supports_jit_vmap_grad():
    x = jnp.asarray([[1.1, 0.3], [0.8, 0.6]], dtype=jnp.float32)
    targets = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    weight_batch = jnp.asarray(
        [
            [[0.6, -0.1], [0.2, 0.4]],
            [[0.4, 0.2], [-0.3, 0.5]],
        ],
        dtype=jnp.float32,
    )

    def loss_for_weight(weight_matrix: jax.Array) -> jax.Array:
        return jax_surrogate_loss(
            [weight_matrix],
            x,
            targets,
            n_steps=4,
            beta=6.0,
            surrogate_path="custom_vjp",
        )

    grad_fn = jax.jit(jax.vmap(jax.grad(loss_for_weight)))
    gradients = grad_fn(weight_batch)

    assert gradients.shape == weight_batch.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_gradient_step_supports_both_paths():
    x = jnp.asarray([[1.0, 0.2], [0.3, 1.1]], dtype=jnp.float32)
    targets = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    initial = [jnp.asarray([[0.5, -0.2], [0.1, 0.4]], dtype=jnp.float32)]

    for path in JAX_SURROGATE_PATHS:
        updated, loss_value = jax_surrogate_gradient_step(
            initial,
            x,
            targets,
            n_steps=5,
            lr=1e-2,
            beta=7.0,
            surrogate_path=path,
        )
        assert len(updated) == 1
        assert updated[0].shape == initial[0].shape
        assert np.isfinite(np.asarray(updated[0])).all()
        assert np.isfinite(loss_value)
