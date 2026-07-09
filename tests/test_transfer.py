# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.transfer
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pytest
from sc_neurocore.transfer import (
    TransferConfig,
    SNNCheckpoint,
    apply_transfer_config,
    freeze_layers,
    load_checkpoint,
    save_checkpoint,
    unfreeze_layers,
)


def _make_checkpoint() -> SNNCheckpoint:
    rng = np.random.default_rng(42)
    return SNNCheckpoint(
        weights=[
            rng.normal(size=(32, 64)).astype(np.float64),
            rng.normal(size=(10, 32)).astype(np.float64),
        ],
        layer_names=["hidden", "output"],
        layer_sizes=[(64, 32), (32, 10)],
        neuron_types=["LIF", "LIF"],
        metadata={"task": "mnist", "accuracy": 0.95},
    )


def _write_minimal_checkpoint(
    path: Path,
    weight: NDArray[np.float64] | None = None,
    *,
    layer_sizes: list[list[int]] | None = None,
) -> None:
    archive_weight = np.array([[1.0]], dtype=np.float64) if weight is None else weight
    np.savez_compressed(str(path) + ".npz", layer_0=archive_weight)
    meta = {
        "layer_names": ["hidden"],
        "layer_sizes": [[1, 1]] if layer_sizes is None else layer_sizes,
        "neuron_types": ["LIF"],
        "frozen_layers": [],
        "n_layers": 1,
        "total_params": int(archive_weight.size),
        "metadata": {},
    }
    Path(str(path) + ".json").write_text(json.dumps(meta), encoding="utf-8")


class TestSNNCheckpoint:
    def test_fields(self) -> None:
        c = _make_checkpoint()
        assert c.n_layers == 2
        assert c.total_params == 32 * 64 + 10 * 32

    def test_rejects_duplicate_layer_names(self) -> None:
        with pytest.raises(ValueError, match="layer_names must be unique"):
            SNNCheckpoint(
                weights=[np.ones((1, 1)), np.ones((1, 1))],
                layer_names=["hidden", "hidden"],
                layer_sizes=[(1, 1), (1, 1)],
            )

    def test_rejects_weight_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape must match layer_sizes"):
            SNNCheckpoint(
                weights=[np.ones((3, 2), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(3, 2)],
            )

    def test_rejects_weight_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="weights length must match"):
            SNNCheckpoint(
                weights=[],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
            )

    def test_rejects_layer_size_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="layer_sizes length must match"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[],
            )

    def test_rejects_neuron_type_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="neuron_types length must match"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
                neuron_types=["LIF", "ALIF"],
            )

    def test_rejects_non_finite_weights(self) -> None:
        with pytest.raises(ValueError, match="finite numeric"):
            SNNCheckpoint(
                weights=[np.array([[np.nan]], dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
            )

    def test_rejects_unknown_frozen_layer(self) -> None:
        with pytest.raises(ValueError, match="frozen_layers must reference known layers"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
                frozen_layers=["missing"],
            )

    def test_rejects_non_string_layer_name(self) -> None:
        with pytest.raises(ValueError, match="layer_names must be a list of strings"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=cast(list[str], ["hidden", 1]),
                layer_sizes=[(1, 1)],
            )

    def test_rejects_boolean_layer_size(self) -> None:
        with pytest.raises(ValueError, match="layer_sizes entries must be integer pairs"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[cast(tuple[int, int], (1, True))],
            )

    def test_rejects_negative_layer_size_constructor(self) -> None:
        with pytest.raises(ValueError, match="layer_sizes entries must be non-negative"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(1, -1)],
            )

    def test_rejects_object_weight_constructor(self) -> None:
        with pytest.raises(ValueError, match="must not contain Python objects"):
            SNNCheckpoint(
                weights=[cast(NDArray[np.float64], np.array([[{"bad": "object"}]], dtype=object))],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
            )

    def test_rejects_string_weight_constructor(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            SNNCheckpoint(
                weights=[cast(NDArray[np.float64], np.array([["bad"]], dtype=np.str_))],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
            )

    def test_rejects_one_dimensional_weight_constructor(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            SNNCheckpoint(
                weights=[cast(NDArray[np.float64], np.array([1.0], dtype=np.float64))],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
            )

    def test_rejects_non_serializable_metadata(self) -> None:
        with pytest.raises(ValueError, match="metadata must be JSON serializable"):
            SNNCheckpoint(
                weights=[np.ones((1, 1), dtype=np.float64)],
                layer_names=["hidden"],
                layer_sizes=[(1, 1)],
                metadata={"bad": {1, 2}},
            )


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.n_layers == 2
        assert loaded.layer_names == ["hidden", "output"]
        assert loaded.metadata["task"] == "mnist"
        np.testing.assert_array_almost_equal(loaded.weights[0], c.weights[0])
        np.testing.assert_array_almost_equal(loaded.weights[1], c.weights[1])

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "nested" / "model"
        save_checkpoint(c, path)
        assert (tmp_path / "nested" / "model.npz").is_file()
        assert (tmp_path / "nested" / "model.json").is_file()

    def test_layer_sizes_preserved(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.layer_sizes == [(64, 32), (32, 10)]

    def test_frozen_layers_preserved(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["hidden"]
        path = tmp_path / "model"
        save_checkpoint(c, path)
        loaded = load_checkpoint(path)
        assert loaded.frozen_layers == ["hidden"]

    def test_rejects_unexpected_archive_members(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_archive_keys"
        np.savez_compressed(
            str(path) + ".npz",
            layer_0=np.array([[1.0]], dtype=np.float64),
            layer_1=np.array([[2.0]], dtype=np.float64),
        )
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1, 1]],
            "neuron_types": ["LIF"],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "bad_archive_keys.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="archive does not match"):
            load_checkpoint(path)

    def test_rejects_total_params_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_total_params"
        _write_minimal_checkpoint(path)
        meta_path = tmp_path / "bad_total_params.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["total_params"] = 2
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="total_params does not match"):
            load_checkpoint(path)

    def test_rejects_metadata_layer_count_mismatch(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        meta_path = tmp_path / "model.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["layer_names"] = ["hidden"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="layer_names length does not match"):
            load_checkpoint(path)

    def test_rejects_negative_layer_size(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        meta_path = tmp_path / "model.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["layer_sizes"] = [[64, -1], [32, 10]]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="must be non-negative"):
            load_checkpoint(path)

    def test_rejects_layer_sizes_length_mismatch(self, tmp_path: Path) -> None:
        c = _make_checkpoint()
        path = tmp_path / "model"
        save_checkpoint(c, path)
        meta_path = tmp_path / "model.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["layer_sizes"] = [[64, 32]]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="layer_sizes length does not match"):
            load_checkpoint(path)

    def test_rejects_malformed_layer_size_pair(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_layer_size_pair"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float64))
        meta = {
            "layer_names": ["hidden"],
            "layer_sizes": [[1]],
            "neuron_types": ["LIF"],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "bad_layer_size_pair.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="integer pairs"):
            load_checkpoint(path)

    def test_rejects_object_dtype_weights(self, tmp_path: Path) -> None:
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

    def test_rejects_non_finite_archive_weight(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_nonfinite"
        _write_minimal_checkpoint(path, np.array([[np.inf]], dtype=np.float64))
        with pytest.raises(ValueError, match="finite numeric"):
            load_checkpoint(path)

    def test_rejects_archive_weight_shape_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_shape"
        _write_minimal_checkpoint(path, np.ones((2, 2), dtype=np.float64), layer_sizes=[[3, 3]])
        with pytest.raises(ValueError, match="shape must match layer_sizes"):
            load_checkpoint(path)

    def test_rejects_non_object_metadata_root(self, tmp_path: Path) -> None:
        path = tmp_path / "badroot"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float32))
        (tmp_path / "badroot.json").write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_checkpoint(path)

    def test_rejects_non_string_layer_names_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_layer_names"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float64))
        meta = {
            "layer_names": ["hidden", 1],
            "layer_sizes": [[1, 1]],
            "neuron_types": [],
            "frozen_layers": [],
            "n_layers": 1,
            "total_params": 1,
            "metadata": {},
        }
        (tmp_path / "bad_layer_names.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="layer_names must be a list of strings"):
            load_checkpoint(path)

    def test_rejects_non_list_layer_sizes(self, tmp_path: Path) -> None:
        path = tmp_path / "badlayers"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float32))
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

    def test_rejects_non_string_frozen_layers(self, tmp_path: Path) -> None:
        path = tmp_path / "badfrozen"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float32))
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

    def test_rejects_boolean_n_layers(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_n_layers_bool"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float32))
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

    def test_rejects_neuron_types_length_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_neuron_types"
        np.savez_compressed(
            str(path) + ".npz",
            layer_0=np.array([[1.0]], dtype=np.float32),
            layer_1=np.array([[2.0]], dtype=np.float32),
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

    def test_rejects_non_object_metadata_field(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_metadata_field"
        np.savez_compressed(str(path) + ".npz", layer_0=np.array([[1.0]], dtype=np.float32))
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

    def test_rejects_invalid_total_params_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_total_params_type"
        _write_minimal_checkpoint(path)
        meta_path = tmp_path / "bad_total_params_type.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["total_params"] = True
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="total_params must be a non-negative integer"):
            load_checkpoint(path)


class TestFreeze:
    def test_freeze_by_name(self) -> None:
        c = _make_checkpoint()
        freeze_layers(c, layer_names=["hidden"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_freeze_until_index(self) -> None:
        c = _make_checkpoint()
        freeze_layers(c, until_index=0)
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_freeze_rejects_unknown_layer(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Unknown layer names"):
            freeze_layers(c, layer_names=["missing"])

    def test_freeze_rejects_non_string_layer_name(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Layer names must be strings"):
            freeze_layers(c, layer_names=cast(Sequence[str], ["hidden", 1]))

    def test_freeze_rejects_negative_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index"):
            freeze_layers(c, until_index=-2)

    def test_freeze_rejects_boolean_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index must be an integer"):
            freeze_layers(c, until_index=cast(int, True))

    def test_freeze_rejects_out_of_range_until_index(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="until_index"):
            freeze_layers(c, until_index=2)

    def test_unfreeze_specific(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, layer_names=["output"])
        assert "hidden" in c.frozen_layers
        assert "output" not in c.frozen_layers

    def test_unfreeze_rejects_unknown_layer(self) -> None:
        c = _make_checkpoint()
        with pytest.raises(ValueError, match="Unknown layer names"):
            unfreeze_layers(c, layer_names=["missing"])

    def test_unfreeze_all(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["hidden", "output"]
        unfreeze_layers(c, all_layers=True)
        assert c.frozen_layers == []

    def test_freeze_layers_deduplicates_and_sorts(self) -> None:
        c = _make_checkpoint()
        c.frozen_layers = ["output"]
        freeze_layers(c, layer_names=["hidden", "output", "hidden"])
        assert c.frozen_layers == ["hidden", "output"]


class TestTransferConfig:
    def test_default_config_is_valid(self) -> None:
        config = TransferConfig()
        assert config.freeze_until == -1
        assert config.lr_backbone == 0.0
        assert config.lr_head == 0.01

    def test_rejects_bool_freeze_until(self) -> None:
        with pytest.raises(ValueError, match="freeze_until"):
            TransferConfig(freeze_until=True)

    def test_rejects_negative_freeze_until_below_sentinel(self) -> None:
        with pytest.raises(ValueError, match="freeze_until index"):
            TransferConfig(freeze_until=-2)

    def test_rejects_non_finite_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_head=np.inf)

    def test_rejects_negative_learning_rate(self) -> None:
        with pytest.raises(ValueError, match="learning rates"):
            TransferConfig(lr_backbone=-0.1)

    def test_apply(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=0, lr_backbone=0.0, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert lrs[0] == 0.0
        assert lrs[1] == 0.01

    def test_apply_by_name(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="hidden", lr_head=0.005)
        c, lrs = apply_transfer_config(c, config)
        assert "hidden" in c.frozen_layers
        assert lrs[1] == 0.005

    def test_no_freeze(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until=-1, lr_head=0.01)
        c, lrs = apply_transfer_config(c, config)
        assert all(lr == 0.01 for lr in lrs)

    def test_apply_by_unknown_name_rejects_config(self) -> None:
        c = _make_checkpoint()
        config = TransferConfig(freeze_until="missing-layer", lr_backbone=0.0, lr_head=0.02)
        with pytest.raises(ValueError, match="freeze_until"):
            apply_transfer_config(c, config)
