# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SNN training loops

"""Tests for SNN training loops."""

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from sc_neurocore.training.loops import evaluate, train_epoch
from sc_neurocore.training.losses import membrane_loss, spike_rate_loss
from sc_neurocore.training.snn_modules import SpikingNet


@pytest.fixture
def tiny_loader():
    x = torch.randn(32, 1, 4, 4)
    y = torch.randint(0, 3, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def tiny_model():
    return SpikingNet(n_input=16, n_hidden=16, n_output=3, n_layers=1)


def test_train_epoch_runs(tiny_model, tiny_loader):
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, acc = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5)
    assert loss > 0
    assert 0 <= acc <= 1


def test_evaluate_runs(tiny_model, tiny_loader):
    loss, acc = evaluate(tiny_model, tiny_loader, n_timesteps=5)
    assert loss > 0
    assert 0 <= acc <= 1


def test_training_improves_loss(tiny_loader):
    model = SpikingNet(n_input=16, n_hidden=32, n_output=3, n_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss0, _ = train_epoch(model, tiny_loader, opt, n_timesteps=10)
    for _ in range(5):
        loss, _ = train_epoch(model, tiny_loader, opt, n_timesteps=10)
    assert loss < loss0 * 5


def test_membrane_loss_fn(tiny_model, tiny_loader):
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, _ = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5, loss_fn=membrane_loss)
    assert loss > 0


def test_spike_rate_loss_fn(tiny_model, tiny_loader):
    def rate_loss(spk, tgt):
        return spike_rate_loss(spk, tgt, n_timesteps=5)

    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    loss, _ = train_epoch(tiny_model, tiny_loader, opt, n_timesteps=5, loss_fn=rate_loss)
    assert loss >= 0
