# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCollectiveFields from former test_swarm_fitness_contracts.py

"""Focused suite: TestCollectiveFields from former test_swarm_fitness_contracts.py."""

from __future__ import annotations

from tests.swarm_fitness_contracts_support import *  # noqa: F403


class TestCollectiveFields:
    @pytest.fixture()
    def fields(self):
        return CollectiveFields(FieldConfig(grid_size=10, seed=42), env_width=100, env_height=100)

    def test_init_shapes(self, fields):
        assert fields.chemical_field.shape == (10, 10)
        assert fields.emotional_field.shape == (20, 8)
        assert fields.symbolic_field.shape == (10, 10, 2)

    def test_deposit_chemical(self, fields):
        fields.deposit_chemical(50.0, 50.0, 1.0)
        assert fields.chemical_field.sum() > 0

    def test_deposit_chemical_negative_ignored(self, fields):
        fields.deposit_chemical(50.0, 50.0, -1.0)
        assert fields.chemical_field.sum() == 0

    def test_diffuse(self, fields):
        fields.deposit_chemical(50.0, 50.0, 10.0)
        before = fields.chemical_field.copy()
        fields.diffuse(dt=1.0)
        assert not np.array_equal(before, fields.chemical_field)

    def test_get_chemical_gradient(self, fields):
        fields.deposit_chemical(60.0, 50.0, 10.0)
        gx, gy = fields.get_chemical_gradient(50.0, 50.0)
        assert isinstance(gx, float)

    def test_synchronize_emotions(self, fields):
        fields.emotional_field[0] = 1.0
        fields.synchronize_emotions()
        assert fields.emotional_field[0, 0] < 1.0

    def test_synchronize_emotions_custom_coupling(self, fields):
        fields.emotional_field[0] = 1.0
        fields.synchronize_emotions(coupling=0.5)
        assert fields.emotional_field[0, 0] < 1.0

    def test_symbolic_deposit_and_read(self, fields):
        fields.deposit_symbolic(25.0, 25.0, 0, 5.0)
        val = fields.get_symbolic_at(25.0, 25.0)
        assert val[0] == 5.0

    def test_apply_laplacian(self):
        field = np.zeros((5, 5))
        field[2, 2] = 1.0
        lap = _apply_laplacian(field)
        assert lap[2, 2] < 0
        assert lap[1, 2] > 0
