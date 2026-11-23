# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusion from former test_fusion.py

"""Focused suite: TestMultiModalFusion from former test_fusion.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.fusion.multimodal import (
    ModalityConfig,
    MultiModalFusion,
)


class TestMultiModalFusion:
    """Fusing several spike modalities."""

    def _make_modalities(self) -> list[ModalityConfig]:
        return [
            ModalityConfig("dvs", n_channels=4, dt_us=100.0),
            ModalityConfig("audio", n_channels=2, dt_us=500.0),
        ]

    def test_concatenate_shape(self) -> None:
        """Concatenation joins the channels of every modality."""
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="concatenate")
        spikes = {
            "dvs": np.random.randint(0, 2, (10, 4)),
            "audio": np.random.randint(0, 2, (10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)

    def test_sum_mode(self) -> None:
        """Sum mode keeps the widest modality and a bounded rate."""
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="sum")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 4)
        assert out.max() <= 1.0

    def test_attention_mode(self) -> None:
        """Attention mode joins the channels of every modality."""
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="attention")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)

    def test_missing_modality(self) -> None:
        """A missing modality contributes zeros."""
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=100.0, mode="concatenate")
        spikes = {"dvs": np.ones((10, 4))}
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape == (10, 6)
        assert np.all(out[:, 4:] == 0)

    def test_resampling(self) -> None:
        """Inputs are resampled to the output timebase."""
        mods = self._make_modalities()
        fuser = MultiModalFusion(mods, output_dt_us=200.0, mode="concatenate")
        spikes = {
            "dvs": np.ones((10, 4)),
            "audio": np.ones((10, 2)),
        }
        out = fuser.fuse(spikes, duration_us=1000.0)
        assert out.shape[0] == 5

    def test_invalid_mode(self) -> None:
        """An unknown fusion mode is rejected."""
        with pytest.raises(ValueError, match="Unknown mode"):
            MultiModalFusion(self._make_modalities(), mode="bad")
