# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrainingLoops from former test_torch_training.py

"""Focused suite: TestTrainingLoops from former test_torch_training.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import pytest
from tests.torch_training_support import (
    SpikingNet,
    auto_device,
    evaluate,
    torch,
    train_epoch,
    training_loops,
)


class TestTrainingLoops:
    """Device selection, training and evaluation loops."""

    @pytest.fixture
    def tiny_loader(self) -> DataLoader[Any]:
        """Return a loader of 32 random samples with five classes in batches of eight."""
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.rand(32, 16)
        y = torch.randint(0, 5, (32,))
        return DataLoader(TensorDataset(x, y), batch_size=8)

    @pytest.fixture
    def tiny_model(self) -> SpikingNet:
        """Return a one-layer spiking network from 16 inputs to five outputs."""
        return SpikingNet(n_input=16, n_hidden=32, n_output=5, n_layers=1)

    def test_auto_device(self) -> None:
        """The automatic device is a torch device."""
        dev = auto_device()
        assert isinstance(dev, torch.device)

    def test_auto_device_falls_back_when_cuda_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing CUDA probe falls back to the CPU."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: device.type != "cuda")
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)

        assert training_loops.auto_device().type == "cpu"

    def test_auto_device_uses_cuda_when_probe_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A passing CUDA probe selects CUDA."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(training_loops, "_cuda_device_supported", lambda index: True)
        monkeypatch.setattr(training_loops, "_device_usable", lambda device: True)

        assert training_loops.auto_device().type == "cuda"

    def test_train_epoch(self, tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
        """One epoch returns a float loss and an accuracy inside [0, 1]."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_evaluate(self, tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
        """Evaluation returns a float loss and an accuracy inside [0, 1]."""
        loss, acc = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert isinstance(loss, float)
        assert 0 <= acc <= 1

    def test_train_reduces_loss(self, tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
        """Six epochs bring the loss below its first-epoch value."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
        loss_0, _ = train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        for _ in range(5):
            train_epoch(tiny_model, tiny_loader, optimizer, n_timesteps=5)
        loss_5, _ = evaluate(tiny_model, tiny_loader, n_timesteps=5)
        assert loss_5 < loss_0

    def test_grad_clipping(self, tiny_model: SpikingNet, tiny_loader: DataLoader[Any]) -> None:
        """An epoch with gradient clipping returns a float loss."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss, acc = train_epoch(
            tiny_model, tiny_loader, optimizer, n_timesteps=5, max_grad_norm=1.0
        )
        assert isinstance(loss, float)
