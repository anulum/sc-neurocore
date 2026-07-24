# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusionSum from former test_fusion_multimodal.py

"""Focused suite: TestMultiModalFusionSum from former test_fusion_multimodal.py."""

from __future__ import annotations

from tests.fusion_multimodal_support import *  # noqa: F403


class TestMultiModalFusionSum:
    def test_sum_mode(self):
        mods = [
            ModalityConfig(name="a", n_channels=3, dt_us=1000.0),
            ModalityConfig(name="b", n_channels=3, dt_us=1000.0),
        ]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="sum")
        assert f.n_output == 3
        trains = {
            "a": np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64),
            "b": np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float64),
        }
        out = f.fuse(trains, duration_us=2000.0)
        assert out.shape == (2, 3)
        assert np.all(out <= 1.0)
        assert np.all(out >= 0.0)

    def test_sum_different_channel_counts(self):
        mods = [
            ModalityConfig(name="a", n_channels=2, dt_us=1000.0),
            ModalityConfig(name="b", n_channels=4, dt_us=1000.0),
        ]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="sum")
        assert f.n_output == 4
        trains = {
            "a": np.ones((3, 2)),
            "b": np.ones((3, 4)),
        }
        out = f.fuse(trains, duration_us=3000.0)
        assert out.shape == (3, 4)
