# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — JAX JIT training demo — surrogate-gradient SNN on

"""
JAX JIT training demo — surrogate-gradient SNN on synthetic data.

Demonstrates jax_forward_pass() and jax_surrogate_gradient_step()
from the JAX backend. Requires JAX: pip install jax jaxlib
"""

from sc_neurocore.accel.jax_backend import (
    HAS_JAX,
    jax_forward_pass,
    jax_surrogate_gradient_step,
)

if not HAS_JAX:
    raise SystemExit("JAX not installed. Run: pip install jax jaxlib")

import jax
import jax.numpy as jnp


def main():
    key = jax.random.PRNGKey(42)
    n_classes = 10
    n_inputs = 64
    n_hidden = 128
    n_steps = 25
    n_epochs = 20
    batch_size = 64
    lr = 5e-3

    # Synthetic classification data
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x_train = jax.random.uniform(k1, (256, n_inputs))
    y_labels = jax.random.randint(k2, (256,), 0, n_classes)
    y_train = jax.nn.one_hot(y_labels, n_classes)

    # Two-layer SNN weights
    weights = [
        jax.random.normal(k3, (n_hidden, n_inputs)) * 0.1,
        jax.random.normal(k4, (n_classes, n_hidden)) * 0.1,
    ]

    # Forward pass demo
    all_spikes, v_final = jax_forward_pass(weights, x_train[:8], n_steps)
    print(f"Forward pass: {len(all_spikes)} layers, output shape {v_final.shape}")

    # Training loop
    for epoch in range(n_epochs):
        idx = jnp.arange(batch_size) % 256
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        weights, loss = jax_surrogate_gradient_step(
            weights,
            x_batch,
            y_batch,
            n_steps=n_steps,
            lr=lr,
        )
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}  loss={loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
