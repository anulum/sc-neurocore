# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCollectiveFields from former test_swarm_control.py

"""Focused suite: TestCollectiveFields from former test_swarm_control.py."""

from __future__ import annotations

from tests.swarm_control_support import *  # noqa: F403

class TestCollectiveFields(unittest.TestCase):
    def test_init(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        self.assertEqual(f.chemical_field.shape, (50, 50))

    def test_deposit_chemical(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        f.deposit_chemical(25.0, 25.0, 1.0)
        self.assertGreater(f.chemical_field.max(), 0)

    def test_diffuse(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=5)
        # Deposit at a valid position (mapped to grid coords internally)
        f.deposit_chemical(50.0, 50.0, 10.0)
        val_before = f.chemical_field.max()
        self.assertGreater(val_before, 0)
        f.diffuse(1.0)
        # After diffusion + decay the peak should decrease
        val_after = f.chemical_field.max()
        self.assertLessEqual(val_after, val_before)

    def test_gradient(self):
        f = CollectiveFields(FieldConfig(grid_size=50), n_agents=5)
        f.deposit_chemical(25.0, 25.0, 10.0)
        gx, gy = f.get_chemical_gradient(24.0, 25.0)
        # Gradient should return floats
        self.assertIsInstance(gx, float)

    def test_emotional_field_shape(self):
        f = CollectiveFields(FieldConfig(), n_agents=10)
        self.assertEqual(f.emotional_field.shape[0], 10)

    def test_synchronize_emotions(self):
        f = CollectiveFields(FieldConfig(), n_agents=5)
        f.emotional_field = np.random.randn(5, 8)
        before_var = f.emotional_field.var()
        f.synchronize_emotions(coupling=0.5)
        after_var = f.emotional_field.var()
        self.assertLessEqual(after_var, before_var + 0.01)

    def test_symbolic_field(self):
        f = CollectiveFields(FieldConfig(grid_size=20), n_agents=3)
        glyph = f.get_symbolic_at(10.0, 10.0)
        self.assertEqual(len(glyph), 2)
