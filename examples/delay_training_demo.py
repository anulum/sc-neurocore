#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- Learnable delay training demo
#
# Usage:
#   pip install sc-neurocore torch
#   PYTHONPATH=src python examples/delay_training_demo.py

"""Train a spiking network with learnable per-synapse delays."""

from __future__ import annotations

import torch

from sc_neurocore.training.delay_linear import DelayLinear
from sc_neurocore.training.surrogate import atan_surrogate


def main():
    print("Learnable Delay Training Demo")
    print("=" * 50)

    torch.manual_seed(42)

    # 1. Create delayed layer
    delay_layer = DelayLinear(
        in_features=8,
        out_features=4,
        max_delay=12,
        learn_delay=True,
        init_delay=3.5,
    )
    print(f"\n1. DelayLinear: {delay_layer.in_features} -> {delay_layer.out_features}")
    print(f"   Max delay: {delay_layer.max_delay} steps")
    print(f"   Initial delays: {delay_layer.delay[0].data.numpy().round(1)}")

    # 2. Train
    optimizer = torch.optim.Adam(delay_layer.parameters(), lr=0.01)
    print("\n2. Training for 50 epochs...")

    for epoch in range(50):
        delay_layer.reset()
        v = torch.zeros(4)
        total_spikes = torch.zeros(4)

        for t in range(20):
            x = torch.randn(8) * (1.0 if t < 3 else 0.0)
            current = delay_layer.step(x)
            v = 0.9 * v + current
            spike = atan_surrogate(v - 1.0)
            v = v - spike.detach()
            total_spikes = total_spikes + spike

        loss = -total_spikes.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: spikes={total_spikes.sum().item():.1f}")

    # 3. Show learned delays
    print("\n3. Learned delays:")
    print(f"   Float: {delay_layer.delay[0].data.numpy().round(2)}")
    print(f"   Integer: {delay_layer.delays_int[0].numpy()}")

    # 4. Export for hardware
    nir_delays = delay_layer.to_nir_delay_array()
    print(f"\n4. NIR delay array: shape={nir_delays.shape}, dtype={nir_delays.dtype}")

    print("\n" + "=" * 50)
    print("Delay training complete.")


if __name__ == "__main__":
    main()
