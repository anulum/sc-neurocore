# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEndToEnd from former test_torch_training.py

"""Focused suite: TestEndToEnd from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestEndToEnd:
    def test_full_pipeline_gradient(self):
        """Complete pipeline: encode → network → loss → backward."""
        x = torch.rand(8, 16)
        labels = torch.randint(0, 5, (8,))
        spikes = rate_encode(x, n_timesteps=10)

        net = SpikingNet(n_input=16, n_hidden=32, n_output=5)
        spike_counts, mem = net(spikes)
        loss = spike_count_loss(spike_counts, labels)
        loss.backward()

        for p in net.parameters():
            assert p.grad is not None

    def test_all_surrogates_train(self):
        """Verify all surrogate functions produce valid training gradients."""
        for fn in [
            fast_sigmoid,
            superspike,
            atan_surrogate,
            sigmoid_surrogate,
            straight_through,
            triangular,
        ]:
            net = SpikingNet(n_input=8, n_hidden=16, n_output=3, surrogate_fn=fn)
            x = torch.randn(5, 2, 8)
            spike_counts, _ = net(x)
            loss = spike_counts.sum()
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            assert len(grads) > 0, f"No gradients with {fn.__name__}"

    def test_sc_export_after_training(self):
        """Train briefly, then export to SC weights."""
        from torch.utils.data import DataLoader, TensorDataset

        net = SpikingNet(n_input=8, n_hidden=16, n_output=3)
        loader = DataLoader(
            TensorDataset(torch.rand(16, 8), torch.randint(0, 3, (16,))), batch_size=8
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        train_epoch(net, loader, opt, n_timesteps=5)
        sc = net.to_sc_weights()
        for layer in sc:
            assert (layer["weight"] >= 0).all()
            assert (layer["weight"] <= 1).all()
