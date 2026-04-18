# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.transfer
from __future__ import annotations
import numpy as np
from sc_neurocore.transfer import (
    save_checkpoint,
    load_checkpoint,
    SNNCheckpoint,
    freeze_layers,
    unfreeze_layers,
    TransferConfig,
)
from sc_neurocore.transfer.fine_tune import apply_transfer_config


def _make_checkpoint():
    return SNNCheckpoint(
        weights=[np.random.randn(32, 64), np.random.randn(10, 32)],
        layer_names=["hidden", "output"],
        layer_sizes=[(64, 32), (32, 10)],
        neuron_types=["LIF", "LIF"],
        metadata={"task": "mnist", "accuracy": 0.95},
    )


class TestSNNCheckpoint:
    def test_fields(self):
        c = _make_checkpoint()
        assert c.n_layers == 2
        assert c.total_params == 32 * 64 + 10 * 32


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.n_layers == 2
        assert loaded.layer_names == ["hidden", "output"]
        assert loaded.metadata["task"] == "mnist"
        np.testing.assert_array_almost_equal(loaded.weights[0], c.weights[0])
        np.testing.assert_array_almost_equal(loaded.weights[1], c.weights[1])

    def test_layer_sizes_preserved(self, tmp_path):
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.layer_sizes == [(64, 32), (32, 10)]

    def test_frozen_layers_preserved(self, tmp_path):
        c = _make_checkpoint()
        c.frozen_layers = ["hidden"]
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.frozen_layers == ["hidden"]


class TestFreeze:
    def test_freeze_by_name(self):
        c = _make_checkpoint()
        freeze_layers(c, layer_names=["hidden"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_freeze_until_index(self):
        c = _make_checkpoint()
        freeze_layers(c, until_index=0)
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_unfreeze_specific(self):
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, layer_names=["output"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_unfreeze_all(self):
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, all_layers=True)
        assert c.frozen_layers == []


class TestTransferConfig:
    def test_apply(self):
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert lrs[0] == 0.0
        assert lrs[1] == 0.01

    def test_apply_by_name(self):
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="hidden", lr_head=0.005)
        c, lrs = apply_transfer_config(c, config)
        assert "hidden" in c.frozen_layers
        assert lrs[1] == 0.005

    def test_no_freeze(self):
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=-1, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert all(lr == 0.01 for lr in lrs)
