# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMemristorMapper from former test_memristor_mapper.py

"""Focused suite: TestMemristorMapper from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestMemristorMapper:
    def test_map_small_matrix(self) -> None:
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        assert result.total_crossbars == 1
        assert result.total_devices == 64

    def test_map_tiled(self) -> None:
        mapper = MemristorMapper(max_crossbar_size=4, seed=42)
        w = np.random.default_rng(0).random((8, 8))
        result = mapper.map_weights(w)
        assert result.total_crossbars == 4

    def test_map_1d_vector(self) -> None:
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random(16)
        result = mapper.map_weights(w)
        assert result.total_crossbars == 1

    def test_error_stats_present(self) -> None:
        mapper = MemristorMapper(seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert result.mean_rel_error >= 0
        assert result.max_rel_error >= 0

    def test_compensation_luts_generated(self) -> None:
        mapper = MemristorMapper(compensation=CompensationStrategy.LUT, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert len(result.mappings[0].compensation_luts) > 0

    def test_no_compensation(self) -> None:
        mapper = MemristorMapper(compensation=CompensationStrategy.NONE, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert len(result.mappings[0].compensation_luts) == 0

    def test_all_technologies(self) -> None:
        for tech in MemristorTechnology:
            mapper = MemristorMapper(technology=tech, seed=42)
            w = np.random.default_rng(0).random((4, 4))
            result = mapper.map_weights(w)
            assert result.total_devices > 0

    def test_differential_topology(self) -> None:
        mapper = MemristorMapper(topology=CrossbarTopology.DIFFERENTIAL, seed=42)
        w = np.random.default_rng(0).random((4, 4))
        result = mapper.map_weights(w)
        assert result.mappings[0].crossbar.topology == CrossbarTopology.DIFFERENTIAL
        assert result.total_devices == 32
