# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-Aware SC-NAS Engine

"""DARTS-based differentiable NAS for SC bitstream optimization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitstreamCandidate(nn.Module):
    """SC bitstream candidate that injects variance for one stream length."""

    def __init__(self, length: int, lut_cost: float, power_cost: float):
        super().__init__()
        self.length = length
        self.lut_cost = lut_cost
        self.power_cost = power_cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the candidate output with training-time SC variance noise."""
        # Simulate the SC variance introduced by limited bitstream length
        # SC variance for independent streams is roughly p*(1-p)/N
        # During training, we inject this as Gaussian noise scaled by the expected variance
        if self.training:
            # We assume x is normalized in [0, 1] probability space
            p = torch.clamp(x, 0.0, 1.0)
            variance = (p * (1.0 - p)) / float(self.length)
            noise = torch.randn_like(x) * torch.sqrt(variance)
            return torch.clamp(x + noise, 0.0, 1.0)
        return x


class SCMixedOp(nn.Module):
    """Continuous relaxation over discrete SC bitstream configurations."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False)

        # Define candidate bitstream lengths
        self.lengths = [64, 128, 256, 512, 1024, 2048, 4096]
        self.num_ops = len(self.lengths)

        # Alpha parameters (architecture weights)
        self.alphas = nn.Parameter(1e-3 * torch.randn(self.num_ops))

        # Instantiate candidate operations
        self.ops = nn.ModuleList()
        macs = float(c_in * c_out * kernel_size * kernel_size)

        for length in self.lengths:
            # Hardware model: LUTs scale with MACs and log(length) for popcounts
            lut_cost = macs * 2.0 + (math.log2(length) * 5.0)
            power_cost = macs * 0.01 * (length / 256.0)
            self.ops.append(BitstreamCandidate(length, lut_cost, power_cost))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mixed convolution output under DARTS bitstream weights."""
        # Compute the baseline conv operation (assumes inputs are probabilities)
        conv_out = self.conv(x)
        # Apply Gumbel-Softmax for differentiable, discrete selection during forward
        weights = F.gumbel_softmax(self.alphas, tau=1.0, hard=False)

        mixed: torch.Tensor = sum(w * op(conv_out) for w, op in zip(weights, self.ops))
        return mixed

    def expected_resource_cost(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expected LUT and power costs from architecture weights."""
        # Expected LUT and Power costs based on current architecture weights
        weights = F.softmax(self.alphas, dim=0)
        exp_luts = sum(w * op.lut_cost for w, op in zip(weights, self.ops))
        exp_power = sum(w * op.power_cost for w, op in zip(weights, self.ops))
        return exp_luts, exp_power

    def extract_optimal_config(self) -> int:
        """Return the bitstream length with the largest architecture logit."""
        idx = int(torch.argmax(self.alphas).item())
        return self.lengths[idx]


class SCNASNetwork(nn.Module):
    """Small differentiable hardware-aware search network for SC-NAS."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = SCMixedOp(1, 16, 3, 1, 1)
        self.layer2 = SCMixedOp(16, 32, 3, 2, 1)
        self.layer3 = SCMixedOp(32, 64, 3, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits from the differentiable SC-NAS network."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits: torch.Tensor = self.fc(x)
        return logits

    def hardware_penalty(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expected LUT and power penalties across search layers."""
        l1, p1 = self.layer1.expected_resource_cost()
        l2, p2 = self.layer2.expected_resource_cost()
        l3, p3 = self.layer3.expected_resource_cost()
        return l1 + l2 + l3, p1 + p2 + p3


if __name__ == "__main__":  # pragma: no cover
    net = SCNASNetwork()
    x = torch.rand(4, 1, 28, 28)

    # 1. Forward Pass (simulating SC variance)
    out = net(x)

    # 2. Extract Differentiable Hardware Costs
    total_luts, total_power = net.hardware_penalty()

    # 3. Compute joint Loss (CrossEntropy + Hardware Constraint Penalty)
    # The gradient will flow through Gumbel-Softmax to update Alpha probabilities,
    # pushing the network towards shorter bitstreams if the penalty weight is high.
    target = torch.randint(0, 10, (4,))
    ce_loss = F.cross_entropy(out, target)

    target_luts = 500_000.0
    hw_loss = torch.relu(total_luts - target_luts) * 1e-4

    loss: torch.Tensor = ce_loss + hw_loss
    loss.backward()

    print("--- SC-NAS Differentiable Search (DARTS) ---")
    print(f"Total Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, HW: {hw_loss.item():.4f})")
    print(f"Expected LUTs: {total_luts.item():.2f}")
    print(f"Expected Power: {total_power.item():.2f} mW")
    assert net.layer1.alphas.grad is not None, (
        "alpha grads should have been populated by backward()"
    )
    print(f"Layer 1 Alphas Grad: {net.layer1.alphas.grad.norm().item():.4f}")

    # Display the optimal extracted architecture
    print("\nOptimal Extracted Bitstream Configurations:")
    print(f"L1: N={net.layer1.extract_optimal_config()}")
    print(f"L2: N={net.layer2.extract_optimal_config()}")
    print(f"L3: N={net.layer3.extract_optimal_config()}")
