# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSParameters from former test_model_golomb_fs.py

"""Focused suite: TestGFSParameters from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403

class TestGFSParameters:
    @pytest.mark.parametrize("g_kv3", [0.0, 150.0, 300.0])
    def test_g_kv3_sweep(self, g_kv3: float):
        n = GolombFSNeuron(g_kv3=g_kv3)
        for _ in range(2000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_na", [50.0, 112.5, 200.0])
    def test_g_na_sweep(self, g_na: float):
        n = GolombFSNeuron(g_na=g_na)
        for _ in range(2000):
            n.step(5.0)
        assert np.isfinite(n.v)
