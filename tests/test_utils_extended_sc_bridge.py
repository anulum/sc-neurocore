# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCBridge from former test_utils_extended.py

"""Focused suite: TestSCBridge from former test_utils_extended.py."""

from __future__ import annotations

from typing import Any
import pytest
import numpy as np
from sc_neurocore.utils.model_bridge import (
    SCBridge,
)


class TestSCBridge:
    """Moving weights between state dictionaries and SC layers."""

    def _make_mock_layer(self, shape: tuple[int, int]) -> Any:
        """Create a simple mock layer with weights attribute."""

        class MockLayer:
            def __init__(self, shape: tuple[int, int]) -> None:
                self.weights = np.zeros(shape)

        return MockLayer(shape)

    def test_load_matching_shapes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Matching weights load normalised to [0, 1]."""
        layer = self._make_mock_layer((3, 4))
        state_dict = {"fc1.weight": np.random.randn(3, 4)}
        layer_mapping = {"fc1": layer}
        SCBridge.load_from_state_dict(state_dict, layer_mapping)
        # Weights should be normalized to [0, 1]
        assert layer.weights.min() >= 0.0
        assert layer.weights.max() <= 1.0

    def test_load_shape_mismatch(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mismatched weights are reported and not loaded."""
        layer = self._make_mock_layer((3, 4))
        state_dict = {"fc1.weight": np.random.randn(5, 6)}
        layer_mapping = {"fc1": layer}
        with caplog.at_level("WARNING", logger="sc_neurocore.utils.model_bridge"):
            SCBridge.load_from_state_dict(state_dict, layer_mapping)
        assert "Shape mismatch" in caplog.text
        # Weights should NOT have changed
        np.testing.assert_array_equal(layer.weights, np.zeros((3, 4)))

    def test_load_missing_key(self, caplog: pytest.LogCaptureFixture) -> None:
        """A layer without weights in the state is reported."""
        layer = self._make_mock_layer((3, 4))
        state_dict = {"other.weight": np.random.randn(3, 4)}
        layer_mapping = {"fc1": layer}
        with caplog.at_level("DEBUG", logger="sc_neurocore.utils.model_bridge"):
            SCBridge.load_from_state_dict(state_dict, layer_mapping)
        assert "No weights found" in caplog.text

    def test_load_updates_individual_synapses(self) -> None:
        """A learning layer exposing a synapse grid has every synapse weight updated."""
        updated: list[tuple[tuple[int, int], float]] = []

        class MockSynapse:
            def __init__(self, i: int, j: int) -> None:
                self._ij = (i, j)

            def update_weight(self, value: float) -> None:
                updated.append((self._ij, float(value)))

        class LearningLayer:
            def __init__(self, shape: tuple[int, int]) -> None:
                self.weights = np.zeros(shape)
                self.synapses = [
                    [MockSynapse(i, j) for j in range(shape[1])] for i in range(shape[0])
                ]

        layer = LearningLayer((2, 3))
        state_dict = {"fc1.weight": np.random.randn(2, 3)}
        SCBridge.load_from_state_dict(state_dict, {"fc1": layer})
        assert len(updated) == 6
        assert {ij for ij, _ in updated} == {(i, j) for i in range(2) for j in range(3)}

    def test_load_layer_without_weights_attribute(self, caplog: pytest.LogCaptureFixture) -> None:
        """A layer lacking a 'weights' attribute is reported and left untouched."""

        class NoWeightsLayer:
            pass

        state_dict = {"fc1.weight": np.random.randn(3, 4)}
        with caplog.at_level("WARNING", logger="sc_neurocore.utils.model_bridge"):
            SCBridge.load_from_state_dict(state_dict, {"fc1": NoWeightsLayer()})
        assert "does not have 'weights'" in caplog.text

    def test_export_to_numpy(self) -> None:
        """Export reads weights from either interface."""

        class MockLayerWithGetWeights:
            def get_weights(self) -> np.ndarray[Any, Any]:
                return np.ones((2, 3))

        class MockLayerWithWeights:
            weights = np.zeros((4, 5))

        layers = {
            "layer_a": MockLayerWithGetWeights(),
            "layer_b": MockLayerWithWeights(),
        }
        state = SCBridge.export_to_numpy(layers)
        assert "layer_a.weight" in state
        assert "layer_b.weight" in state
        np.testing.assert_array_equal(state["layer_a.weight"], np.ones((2, 3)))
        np.testing.assert_array_equal(state["layer_b.weight"], np.zeros((4, 5)))

    def test_export_empty(self) -> None:
        """Exporting no layers gives an empty state."""
        state = SCBridge.export_to_numpy({})
        assert state == {}

    def test_load_triggers_refresh(self, capsys: pytest.CaptureFixture[str]) -> None:
        """If layer has _refresh_packed_weights, it should be called."""
        refreshed = [False]

        class MockLayerRefresh:
            def __init__(self) -> None:
                self.weights = np.zeros((2, 3))

            def _refresh_packed_weights(self) -> None:
                refreshed[0] = True

        layer = MockLayerRefresh()
        state_dict = {"fc1.weight": np.random.randn(2, 3)}
        SCBridge.load_from_state_dict(state_dict, {"fc1": layer})
        assert refreshed[0] is True
