#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pure-numpy SHD inference (no SpikingJelly dependency)
"""SC-NeuroCore reference inference for Masquelier/Queant SHD model.

Implements the exact same dynamics as their SpikingJelly Vmin_LIFNode +
DCLS axonal delays, but using only numpy. This is the step before
Verilog generation — proves we can reproduce their results without
their framework.

Architecture (axonal delays, QAT sparsity 90%):
    Input(140) → AxDelay(140) → Linear(140→128) → Vmin_LIF
               → AxDelay(128) → Linear(128→128) → Vmin_LIF
               → Linear(128→20) → Output (sum membrane over time)

Neuron dynamics (Vmin_LIF, decay_input=False, hard reset):
    v = v * (1 - 1/tau) + x
    spike = (v >= v_threshold)
    v = v_reset * spike + (1 - spike) * v
    v = v_inf + softplus(v - v_inf, beta=beta_v_inf)
"""

from __future__ import annotations

import json
import os

import numpy as np


def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(beta*x)) / beta."""
    bx = beta * x
    return np.where(bx > threshold, x, np.log1p(np.exp(bx)) / beta)


class VminLIF:
    """Vmin_LIFNode equivalent in pure numpy."""

    def __init__(
        self,
        n_neurons: int,
        tau: float = 4.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_inf: float = -5.0,
        beta_v_inf: float = 1.0,
    ):
        self.n = n_neurons
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_inf = v_inf
        self.beta_v_inf = beta_v_inf
        self.v = np.zeros(n_neurons, dtype=np.float32)

    def reset(self) -> None:
        self.v[:] = 0.0

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single timestep. Returns binary spike array."""
        # Charge (decay_input=False, hard reset)
        self.v = self.v * (1.0 - 1.0 / self.tau) + x

        # Spike
        spike = (self.v >= self.v_threshold).astype(np.float32)

        # Reset
        self.v = self.v_reset * spike + (1.0 - spike) * self.v

        # Voltage floor (softplus clamp)
        self.v = self.v_inf + softplus(self.v - self.v_inf, beta=self.beta_v_inf)

        return spike


class AxonalDelayBuffer:
    """Circular buffer implementing per-neuron axonal delays."""

    def __init__(self, n_neurons: int, delays: np.ndarray, max_delay: int = 31):
        self.n = n_neurons
        self.delays = delays.astype(np.int32)  # integer delays per neuron
        self.max_delay = max_delay
        # Buffer: (max_delay, n_neurons), circular
        self.buffer = np.zeros((max_delay, n_neurons), dtype=np.float32)
        self.write_idx = 0

    def reset(self) -> None:
        self.buffer[:] = 0.0
        self.write_idx = 0

    def step(self, x: np.ndarray) -> np.ndarray:
        """Write new input, read delayed output."""
        # Write current input
        self.buffer[self.write_idx % self.max_delay] = x

        # Read at delay offset for each neuron
        output = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            read_idx = (self.write_idx - self.delays[i]) % self.max_delay
            output[i] = self.buffer[read_idx, i]

        self.write_idx += 1
        return output


class SHDModel:
    """Complete SHD inference model in pure numpy."""

    def __init__(self, export_dir: str):
        with open(os.path.join(export_dir, "config.json")) as f:
            self.config = json.load(f)

        # Load weights
        self.w1 = np.load(os.path.join(export_dir, "layers_1_weight.npy"))  # (128, 140)
        self.w2 = np.load(os.path.join(export_dir, "layers_6_weight.npy"))  # (128, 128)
        self.w3 = np.load(os.path.join(export_dir, "layers_10_weight.npy"))  # (20, 128)

        # Load delays (rounded to int)
        d1 = np.load(os.path.join(export_dir, "layers_0_P_int.npy"))  # (140,) int8
        d2 = np.load(os.path.join(export_dir, "layers_5_P_int.npy"))  # (128,) int8

        # DCLS delay offset: delay buffers use left_padding = max_delay - 1 = 30
        # Positive P means earlier read (less delay), negative means more delay
        # Convert DCLS positions to buffer tap offsets
        # In DCLS: P is position in kernel, range [-15, 15] for max_delay=31
        # Buffer delay = (max_delay - 1) // 2 - P = 15 - P
        self.delay1_offsets = 15 - d1.astype(np.int32)
        self.delay2_offsets = 15 - d2.astype(np.int32)

        # Build layers
        self.delay1 = AxonalDelayBuffer(140, self.delay1_offsets, max_delay=31)
        self.lif1 = VminLIF(128)
        self.delay2 = AxonalDelayBuffer(128, self.delay2_offsets, max_delay=31)
        self.lif2 = VminLIF(128)

    def reset(self) -> None:
        self.delay1.reset()
        self.lif1.reset()
        self.delay2.reset()
        self.lif2.reset()

    def forward_step(self, x: np.ndarray) -> np.ndarray:
        """Single timestep forward pass. Returns output logits (20,)."""
        # Layer 1: delay → linear → LIF
        d1_out = self.delay1.step(x)
        h1 = d1_out @ self.w1.T  # (140,) @ (140, 128).T = (128,)
        s1 = self.lif1.step(h1)

        # Layer 2: delay → linear → LIF
        d2_out = self.delay2.step(s1)
        h2 = d2_out @ self.w2.T  # (128,) @ (128, 128).T = (128,)
        s2 = self.lif2.step(h2)

        # Output: linear (no spike, accumulate membrane)
        out = s2 @ self.w3.T  # (128,) @ (128, 20).T = (20,)
        return out

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """Full sequence forward. x_seq: (T, 140). Returns (T, 20)."""
        self.reset()
        T = x_seq.shape[0]
        outputs = np.zeros((T, 20), dtype=np.float32)
        for t in range(T):
            outputs[t] = self.forward_step(x_seq[t])
        return outputs

    def classify(self, x_seq: np.ndarray) -> int:
        """Classify a single sample. Returns predicted class (0-19)."""
        out = self.forward(x_seq)
        return int(out.sum(axis=0).argmax())


def compare_with_spikingjelly(export_dir: str, n_samples: int = 10) -> None:
    """Compare SC-NeuroCore inference with SpikingJelly reference."""
    import sys
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(export_dir), "neuromorphic_training-main"))
    os.environ["WANDB_MODE"] = "disabled"

    from configs.config_SHD import Config
    from spikingjelly.activation_based import functional
    from src.SHD.snn import SNN_axonal_feedforward_delays

    config = Config()
    config.hidden_layers = [128, 128]

    # Load SpikingJelly model
    ckpt_path = os.path.join(
        os.path.dirname(export_dir),
        "neuromorphic_training-main/exp/SHD/SNN_axonal_feedforward_delays/quantized_sparsity_90/best.pth",
    )
    sj_model = SNN_axonal_feedforward_delays(config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sj_model.load_state_dict(ckpt["net"])
    sj_model.eval()

    # Load SC-NeuroCore model
    sc_model = SHDModel(export_dir)

    # Load test data
    from src.datasets import load_dataset

    _, _, test_loader = load_dataset(config)

    sc_correct = 0
    sj_correct = 0
    match = 0
    total = 0

    for batch_idx, (x, label, *_) in enumerate(test_loader):
        if total >= n_samples:
            break

        x_np = x.numpy()  # (B, T, N)
        x_torch = x.float().permute(1, 0, 2)  # (T, B, N)

        # SpikingJelly
        functional.reset_net(sj_model)
        with torch.no_grad():
            sj_out = sj_model(x_torch)  # (T, B, 20)
        sj_pred = sj_out.sum(0).argmax(1).numpy()  # (B,)

        # SC-NeuroCore (per sample)
        B = x_np.shape[0]
        sc_pred = np.zeros(B, dtype=np.int64)
        for b in range(B):
            sc_pred[b] = sc_model.classify(x_np[b])  # (T, N)

        labels = label.numpy()
        for b in range(B):
            if total >= n_samples:
                break
            sc_ok = sc_pred[b] == labels[b]
            sj_ok = sj_pred[b] == labels[b]
            pred_match = sc_pred[b] == sj_pred[b]
            sc_correct += int(sc_ok)
            sj_correct += int(sj_ok)
            match += int(pred_match)
            total += 1

    print(f"\n=== SC-NeuroCore vs SpikingJelly comparison ({total} samples) ===")
    print(f"  SpikingJelly accuracy: {100 * sj_correct / total:.1f}%")
    print(f"  SC-NeuroCore accuracy: {100 * sc_correct / total:.1f}%")
    print(f"  Prediction match: {100 * match / total:.1f}%")
    print("  (Match means both models predict the same class, regardless of correctness)")


if __name__ == "__main__":
    export_dir = os.path.join(os.path.dirname(__file__), "exported")

    # Quick standalone test
    model = SHDModel(export_dir)
    print(f"Model loaded: {model.config['architecture']}/{model.config['variant']}")
    print(f"Weights: w1={model.w1.shape}, w2={model.w2.shape}, w3={model.w3.shape}")
    print(
        f"Delays: d1={len(model.delay1_offsets)} [{model.delay1_offsets.min()},{model.delay1_offsets.max()}]"
    )
    print(
        f"        d2={len(model.delay2_offsets)} [{model.delay2_offsets.min()},{model.delay2_offsets.max()}]"
    )

    # Test with random input
    T = 100
    x = (np.random.default_rng(42).random((T, 140)) < 0.02).astype(np.float32)
    out = model.forward(x)
    pred = out.sum(axis=0).argmax()
    print(f"\nRandom input ({T} steps, ~2% spike rate): predicted class {pred}")
    print(f"Output sum per class: {out.sum(0)}")

    # Compare with SpikingJelly if available
    try:
        compare_with_spikingjelly(export_dir, n_samples=100)
    except Exception as e:
        print(f"\nSpikingJelly comparison skipped: {e}")
