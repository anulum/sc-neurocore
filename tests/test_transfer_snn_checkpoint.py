# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNNCheckpoint from former test_transfer.py

"""Focused suite: TestSNNCheckpoint from former test_transfer.py."""

from __future__ import annotations

from tests.transfer_support import *  # noqa: F403

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
