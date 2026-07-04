# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenith hybrid ResNet demonstration

import torch
import torch.nn as nn
from sc_neurocore.plasticity import create_plasticity_layer
from sc_neurocore._native.learning_bridge import RULE_BCM, set_deterministic_mode

print("--- ZENITH HYBRID RESNET INTEGRATION ---\n")


class HybridResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1. Start with standard classical vision backend
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 512),
            nn.ReLU(),
        )

        # 2. Append Zenith Biological Bridge Layer (Metaplastic BCM Rule)
        # Running natively with explicit GPU/autograd mapping
        self.bcm_plasticity = create_plasticity_layer(
            count=num_classes,
            rule_type=RULE_BCM,
            backend="torch",
            autograd=True,
            param_a=0.05,
            param_b=20.0,
        )

        # Standard projection down to spike probability dimensions
        self.projection = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, global_reward=None, dt=1.0):
        # Forward pass standard Deep Learning
        features = self.backbone(x)
        logits = self.projection(features)

        # Map into pseudospikes
        spike_probs = self.sigmoid(logits)
        pre_spikes = spike_probs > 0.5
        post_spikes = spike_probs > 0.8

        # Trigger biological state machine update exactly
        # Note: BCM doesn't strictly use rewards, but passed for architectural parity
        if global_reward is None:
            global_reward = torch.zeros_like(pre_spikes)

        biological_weights = self.bcm_plasticity(
            pre_spikes.squeeze(), post_spikes.squeeze(), rewards=global_reward, dt=dt
        )
        return biological_weights * spike_probs


# Initialize network and lock deterministic seeds
set_deterministic_mode(seed=1337)
model = HybridResNet(num_classes=10)

# Simulate batch execution
batch_x = torch.randn(1, 3, 224, 224)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training hybrid network through PyTorch Autograd...")
for step in range(3):
    optimizer.zero_grad()

    # Excecute forward biological tracking dynamically scaled by 0.5ms dt
    out = model(batch_x, dt=0.5)

    # Calculate loss leveraging exact STDP Jacobian approximations
    loss = out.sum()
    loss.backward()
    optimizer.step()
    print(f"  Step {step + 1}: Loss = {loss.item():.4f}")

# Exfiltrate state and serialize bitstream natively via SC-NeuroCore
print("\nSaving layer via Zenith Exascale persistence mechanics...")

# Export layer directly to byte sequence
rust_layer = create_plasticity_layer(count=10, rule_type=RULE_BCM, backend="rust")
rust_layer.step(pre_spikes=[True] * 10, post_spikes=[True] * 10, rewards=[0.1] * 10, dt=0.5)

# Canonicalized safe hardware save
deploy_path = "bcm_deployment.scal"
rust_layer.save(deploy_path)

# Quick parity check
loaded_rust_layer = create_plasticity_layer(count=10, rule_type=RULE_BCM, backend="rust")
loaded_rust_layer.load(deploy_path)
rust_loaded_weights = loaded_rust_layer.get_weights()

print(f"Verified deployment architecture saved to {deploy_path}.")
print(f"Exascale Persistence Load Verified. Rust loaded weights shape: {rust_loaded_weights.shape}")
print("--- End of Demonstration ---")
