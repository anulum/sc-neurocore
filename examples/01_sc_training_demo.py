#!/usr/bin/env python
"""
SC-NeuroCore v3 - Surrogate Gradient Training Demo
==================================================

Demonstrates end-to-end training of a stochastic computing
dense layer using surrogate gradients.

Task: Binary classification on a simple 2D dataset (XOR-like).
The SC layer learns to separate the classes using bitstream
computation with surrogate-gradient-based weight updates.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python examples/01_sc_training_demo.py
"""

from __future__ import annotations

import numpy as np

from sc_neurocore_engine import DifferentiableDenseLayer


def generate_xor_data(n_samples: int, rng: np.random.RandomState):
    """Generate a noisy XOR classification dataset."""
    X = rng.uniform(0, 1, (n_samples, 4))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        a = int(X[i, 0] > 0.5) ^ int(X[i, 1] > 0.5)
        b = int(X[i, 2] > 0.5) ^ int(X[i, 3] > 0.5)
        y[i] = float(a ^ b)
    return X, y


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((pred - target) ** 2))


def mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss w.r.t. predictions."""
    return 2.0 * (pred - target) / len(pred)


def main():
    rng = np.random.RandomState(42)
    n_train = 200
    n_epochs = 50
    lr = 0.005
    length = 2048

    print("SC-NeuroCore v3 - Surrogate Gradient Training Demo")
    print("=" * 55)

    X_train, y_train = generate_xor_data(n_train, rng)

    layer = DifferentiableDenseLayer(
        n_inputs=4,
        n_neurons=1,
        length=length,
        surrogate="fast_sigmoid",
        k=25.0,
    )

    print(f"Layer: 4 -> 1, L={length}, surrogate=fast_sigmoid")
    print(f"Training: {n_train} samples, {n_epochs} epochs, lr={lr}")
    print()
    print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 32)

    for epoch in range(n_epochs):
        total_loss = 0.0
        predictions = []
        targets = []

        for i in range(n_train):
            x = X_train[i]
            target = np.array([y_train[i]])

            pred = layer.forward(x, seed=42 + epoch * n_train + i)
            loss = mse_loss(pred, target)
            total_loss += loss

            predictions.append(float(pred[0]))
            targets.append(float(target[0]))

            grad_out = mse_grad(pred, target)
            _, grad_w = layer.backward(grad_out)
            layer.update_weights(grad_w, lr=lr)

        avg_loss = total_loss / n_train
        correct = sum(
            1 for p, t in zip(predictions, targets) if (p > 0.5) == (t > 0.5)
        )
        accuracy = correct / len(targets) * 100.0

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:<8} {avg_loss:<12.6f} {accuracy:<10.1f}%")

    print()
    print("Training complete.")
    print("Note: SC layers are stochastic - loss may fluctuate.")
    print("The surrogate gradient enables weight updates despite")
    print("the non-differentiable bitstream AND+popcount forward pass.")

    final_preds = [layer.forward(x, seed=9999)[0] for x in X_train]
    final_correct = sum(
        1 for p, t in zip(final_preds, y_train) if (p > 0.5) == (t > 0.5)
    )
    print(
        f"\nFinal accuracy: {final_correct}/{len(y_train)} "
        f"({final_correct/len(y_train):.0%})"
    )


if __name__ == "__main__":
    main()
