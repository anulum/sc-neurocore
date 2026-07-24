# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusion from former test_fusion.py

"""Focused suite: TestMultiModalFusion from former test_fusion.py."""

from __future__ import annotations

from tests.fusion_support import *  # noqa: F403


class TestMultiModalFusion:
    def _make_modalities(self):
        return [
            ModalityConfig("dvs", n_channels=4, dt_us=100.0),
            ModalityConfig("audio", n_channels=2, dt_us=500.0),
        ]

    def test_concatenate_shape(self):
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="concatenate")
        spikes = {
            "dvs": np.random.randint(0, 2, (10, 4)),
            "audio": np.random.randint(0, 2, (10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)

    def test_sum_mode(self):
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="sum")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 4)
        assert out.max() <= 1.0

    def test_attention_mode(self):
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="attention")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)

    def test_missing_modality(self):
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="concatenate")
        spikes = {"dvs": np.ones((10, 4))}
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)
        assert np.all(out[:, 4:] == 0)

    def test_resampling(self):
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=200.0, mode="concatenate")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape[0] == 5

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            MultiModalFusion(self._make_modalities(), mode="bad")
