# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.transfer
from __future__ import annotations
import json
import numpy as np
import pytest
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

    def test_rejects_metadata_layer_count_mismatch(self, tmp_path):
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        meta_path = tmp_path / "model.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["layer_names"] = ["hidden"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="layer_names length does not match"):
            load_checkpoint(path)

    def test_rejects_negative_layer_size(self, tmp_path):
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        meta_path = tmp_path / "model.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["layer_sizes"] = [[64, -1], [32, 10]]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="must be non-negative"):
            load_checkpoint(path)

    def test_rejects_object_dtype_weights(self, tmp_path):
        path = tmp_path / "model"
        np.savez_compressed(
            str(path) + ".npz",
            layer_0=np.array([{"bad": "object"}], dtype=object),
        )
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1, 1]],
            "neuron_types": [],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "model.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="allow_pickle=False"):
            load_checkpoint(path)

    def test_rejects_non_object_metadata_root(self, tmp_path):
        path = tmp_path / "badroot"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([1.0], dtype=np.float32))
        (tmp_path / "badroot.json").write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_checkpoint(path)

    def test_rejects_non_list_layer_sizes(self, tmp_path):
        path = tmp_path / "badlayers"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([1.0], dtype=np.float32))
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": "not-a-list",
            "neuron_types": [],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "badlayers.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="layer_sizes must be a list"):
            load_checkpoint(path)

    def test_rejects_non_string_frozen_layers(self, tmp_path):
        path = tmp_path / "badfrozen"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([1.0], dtype=np.float32))
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1, 1]],
            "neuron_types": [],
            "frozen_layers": [1],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "badfrozen.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="frozen_layers must be a list of strings"):
            load_checkpoint(path)

    def test_rejects_boolean_n_layers(self, tmp_path):
        path = tmp_path / "bad_n_layers_bool"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([1.0], dtype=np.float32))
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1, 1]],
            "neuron_types": [],
            "frozen_layers": [],
            "n_layers": True,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "bad_n_layers_bool.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="n_layers must be a non-negative integer"):
            load_checkpoint(path)

    def test_rejects_neuron_types_length_mismatch(self, tmp_path):
        path = tmp_path / "bad_neuron_types"
        np.savez_compressed(
            str(path) + ".npz",
            layer_0=np.array([1.0], dtype=np.float32),
            layer_1=np.array([2.0], dtype=np.float32),
        )
        meta = {
            "layer_names": ["l0", "l1"],
            "layer_sizes": [[1, 1], [1, 1]],
            "neuron_types": ["LIF"],
            "frozen_layers": [],
            "n_layers": 2,
            "total_params": 2,
            "metadata": {},
        }
        (tmp_path / "bad_neuron_types.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="neuron_types length does not match"):
            load_checkpoint(path)

    def test_rejects_non_object_metadata_field(self, tmp_path):
        path = tmp_path / "bad_metadata_field"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([1.0], dtype=np.float32))
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1, 1]],
            "neuron_types": [],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": [],
        }
        (tmp_path / "bad_metadata_field.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="metadata field must be a JSON object"):
            load_checkpoint(path)


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

    def test_freeze_layers_deduplicates_and_sorts(self):
        c = _make_checkpoint()
        c.frozen_layers = ["output"]
        freeze_layers(c, layer_names=["hidden", "output", "hidden"])
        assert c.frozen_layers == ["hidden", "output"]


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

    def test_apply_by_unknown_name_does_not_freeze_layers(self):
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="missing-layer", lr_backbone=0.0, lr_head=0.02)
        c, lrs = apply_transfer_config(c, config)
        assert c.frozen_layers == []
        assert lrs == [0.02, 0.02]
