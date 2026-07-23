# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusionConcatenate from former test_fusion_multimodal.py

"""Focused suite: TestMultiModalFusionConcatenate from former test_fusion_multimodal.py."""

from __future__ import annotations

from tests.fusion_multimodal_support import *  # noqa: F403

class TestMultiModalFusionConcatenate:
    def _make_fusion(self):
        mods = [
            ModalityConfig(name="dvs", n_channels=4, dt_us=1000.0),
            ModalityConfig(name="audio", n_channels=3, dt_us=1000.0),
        ]
        return MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")

    def test_output_channels(self):
        f = self._make_fusion()
        assert f.n_output == 7

    def test_fuse_same_timebase(self):
        f = self._make_fusion()
        trains = {
            "dvs": np.ones((5, 4)),
            "audio": np.ones((5, 3)),
        }
        out = f.fuse(trains, duration_us=5000.0)
        assert out.shape == (5, 7)

    def test_missing_modality_zeros(self):
        f = self._make_fusion()
        trains = {"dvs": np.ones((5, 4))}
        out = f.fuse(trains, duration_us=5000.0)
        assert out.shape == (5, 7)
        assert np.all(out[:, 4:] == 0)

    def test_resampling(self):
        mods = [
            ModalityConfig(name="fast", n_channels=2, dt_us=500.0),
            ModalityConfig(name="slow", n_channels=2, dt_us=2000.0),
        ]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")
        trains = {
            "fast": np.ones((10, 2)),
            "slow": np.ones((3, 2)),
        }
        out = f.fuse(trains, duration_us=5000.0)
        assert out.shape[0] == 5
        assert out.shape[1] == 4
