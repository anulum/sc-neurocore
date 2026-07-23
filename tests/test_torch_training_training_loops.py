# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrainingLoops from former test_torch_training.py

"""Focused suite: TestTrainingLoops from former test_torch_training.py."""

from __future__ import annotations

from tests.torch_training_support import *  # noqa: F403

class TestTrainingLoops:
    @pytest.fixture
    def tiny_loader(self):
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.rand(32, 16)
        y = torch.randint(0, 5, (32,))
        return DataLoader(TensorDataset(x, y), batch_size=8)

    @pytest.fixture
    def tiny_model(self):
        return SpikingNet(n_input=16, n_hidden=32, n_output=5, n_layers=1)

    def test_auto_device(self):
        dev = auto_device()
        assert isinstance(dev, torch.device)

    def test_auto_device_falls_back_when_cuda_probe_fails(self, monkeypatch):
        monkeypatch.setattr(training_loops.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: device.type != "cuda")
        if hasattr(training_loops.torch.backends, "mps"):
            monkeypatch.setattr(
                training_loops.torch.backends.mps, "is_available", lambda: False, raising=False
            )

        assert training_loops.auto_device().type == "cpu"

    def test_auto_device_uses_cuda_when_probe_passes(self, monkeypatch):
        monkeypatch.setattr(training_loops.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(training_loops.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(training_loops, "_cuda_device_supported", lambda index: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: True)

        assert training_loops.auto_device().type == "cuda"

    def test_train_epoch(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_evaluate(self, tiny_model, tiny_loader):
        loss, acc = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_train_reduces_loss(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
        loss_0, _ = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        for _ in range(5):
            train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        loss_5, _ = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert loss_5 < loss_0

    def test_grad_clipping(self, tiny_model, tiny_loader):
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(
            tiny_model, tiny_loader, optimizer, n_timesteps=5, max_grad_norm=1.0
        )
        assert isinstance(loss, float)
