"""Tests for CollectiveFields."""

import numpy as np
import pytest

from sc_neurocore.swarm.collective_fields import CollectiveFields, FieldConfig


class TestCollectiveFields:
    def test_init(self):
        fields = CollectiveFields()
        assert fields.chemical_field.shape == (50, 50)
        assert fields.symbolic_field.shape == (50, 50, 2)

    def test_custom_config(self):
        cfg = FieldConfig(grid_resolution=20, symbolic_dims=3)
        fields = CollectiveFields(config=cfg)
        assert fields.chemical_field.shape == (20, 20)
        assert fields.symbolic_field.shape == (20, 20, 3)

    def test_deposit_chemical(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10, arena_width=100))
        fields.deposit_chemical(50.0, 50.0, 1.0)
        assert fields.chemical_field.sum() > 0

    def test_chemical_gradient(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=20, arena_width=100))
        # Create a gradient by depositing at one spot
        fields.deposit_chemical(70.0, 50.0, 10.0)
        grad = fields.get_chemical_gradient(60.0, 50.0)
        assert grad.shape == (2,)
        assert np.all(grad >= 0) and np.all(grad <= 1)

    def test_diffuse_chemical(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10))
        fields.chemical_field[5, 5] = 10.0
        fields.diffuse_chemical(dt=1.0)
        # Center should decrease, neighbors should increase
        assert fields.chemical_field[5, 5] < 10.0
        assert fields.chemical_field[5, 6] > 0

    def test_chemical_decay(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10, chem_decay_rate=0.5))
        fields.chemical_field[:] = 5.0
        total_before = fields.chemical_field.sum()
        fields.diffuse_chemical(dt=1.0)
        total_after = fields.chemical_field.sum()
        assert total_after < total_before

    def test_synchronize_emotions(self):
        fields = CollectiveFields(FieldConfig(n_emotional_dims=4))
        emotions = [
            np.array([1.0, 0.0, 0.5, 0.5]),
            np.array([0.0, 1.0, 0.5, 0.5]),
        ]
        updated = fields.synchronize_emotions(emotions, coupling=0.5)
        assert len(updated) == 2
        # After coupling, values should be closer to mean
        assert abs(updated[0][0] - updated[1][0]) < abs(emotions[0][0] - emotions[1][0])

    def test_emotion_mean(self):
        fields = CollectiveFields(FieldConfig(n_emotional_dims=4))
        emotions = [np.array([0.8, 0.2, 0.5, 0.5])]
        fields.synchronize_emotions(emotions)
        mean = fields.get_emotion_mean()
        np.testing.assert_allclose(mean, [0.8, 0.2, 0.5, 0.5], atol=0.01)

    def test_deposit_symbolic(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10))
        fields.deposit_symbolic(50.0, 50.0, np.array([0.5, -0.3]))
        assert fields.symbolic_field.sum() != 0

    def test_get_symbolic_value(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10))
        val = fields.get_symbolic_value(50.0, 50.0)
        assert val.shape == (2,)
        assert np.all(val >= 0) and np.all(val <= 1)

    def test_step_diffuses(self):
        fields = CollectiveFields(FieldConfig(grid_resolution=10))
        fields.chemical_field[5, 5] = 10.0
        fields.step(dt=1.0)
        assert fields.chemical_field[5, 5] < 10.0

    def test_reset(self):
        fields = CollectiveFields()
        fields.chemical_field[0, 0] = 999
        fields.reset()
        assert fields.chemical_field.sum() == 0

    def test_get_state(self):
        fields = CollectiveFields()
        state = fields.get_state()
        assert "chemical_field_sum" in state
        assert "emotion_mean" in state
