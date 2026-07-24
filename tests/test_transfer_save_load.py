# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSaveLoad from former test_transfer.py

"""Focused suite: TestSaveLoad from former test_transfer.py."""

from __future__ import annotations

from tests.transfer_support import *  # noqa: F403


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
