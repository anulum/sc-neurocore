# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.fusion (multimodal spike fusion)

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.fusion.multimodal import MultiModalFusion, ModalityConfig


class TestModalityConfig:
    def test_defaults(self):
        m = ModalityConfig(name="dvs", n_channels=128, dt_us=1000.0)
        assert m.max_rate_hz == 1000.0

    def test_custom(self):
        m = ModalityConfig(name="audio", n_channels=64, dt_us=500.0, max_rate_hz=2000.0)
        assert m.name == "audio"
        assert m.n_channels == 64


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


class TestMultiModalFusionAttention:
    def test_attention_mode(self):
        mods = [
            ModalityConfig(name="dvs", n_channels=3, dt_us=1000.0),
            ModalityConfig(name="imu", n_channels=2, dt_us=1000.0),
        ]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="attention")
        assert f.n_output == 5
        assert len(f.attention_weights) == 2
        assert f.attention_weights.sum() == pytest.approx(1.0)

        trains = {
            "dvs": np.ones((4, 3)),
            "imu": np.ones((4, 2)),
        }
        out = f.fuse(trains, duration_us=4000.0)
        assert out.shape == (4, 5)


class TestMultiModalFusionEdgeCases:
    def test_unknown_mode_raises(self):
        mods = [ModalityConfig(name="x", n_channels=1, dt_us=1000.0)]
        with pytest.raises(ValueError, match="Unknown mode"):
            MultiModalFusion(mods, mode="invalid")

    def test_zero_duration(self):
        mods = [ModalityConfig(name="x", n_channels=2, dt_us=1000.0)]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")
        trains = {"x": np.ones((1, 2))}
        out = f.fuse(trains, duration_us=0.0)
        assert out.shape[0] >= 1

    def test_rate_normalization(self):
        mods = [ModalityConfig(name="x", n_channels=2, dt_us=1000.0)]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")
        trains = {"x": np.array([[5.0, 10.0], [3.0, 7.0]])}
        out = f.fuse(trains, duration_us=2000.0)
        assert out.max() <= 1.0
