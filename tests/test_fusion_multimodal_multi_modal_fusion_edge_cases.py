# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusionEdgeCases from former test_fusion_multimodal.py

"""Focused suite: TestMultiModalFusionEdgeCases from former test_fusion_multimodal.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.fusion.multimodal import (
    ModalityConfig,
    MultiModalFusion,
)


class TestMultiModalFusionEdgeCases:
    """Edge cases of multimodal fusion."""

    def test_unknown_mode_raises(self) -> None:
        """An unknown fusion mode is rejected."""
        mods = [ModalityConfig(name="x", n_channels=1, dt_us=1000.0)]
        with pytest.raises(ValueError, match="Unknown mode"):
            MultiModalFusion(mods, mode="invalid")

    def test_zero_duration(self) -> None:
        """A zero duration still gives one output step."""
        mods = [ModalityConfig(name="x", n_channels=2, dt_us=1000.0)]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")
        trains = {"x": np.ones((1, 2))}
        out = f.fuse(trains, duration_us=0.0)
        assert out.shape[0] >= 1

    def test_rate_normalization(self) -> None:
        """Fused rates stay at or below 1."""
        mods = [ModalityConfig(name="x", n_channels=2, dt_us=1000.0)]
        f = MultiModalFusion(mods, output_dt_us=1000.0, mode="concatenate")
        trains = {"x": np.array([[5.0, 10.0], [3.0, 7.0]])}
        out = f.fuse(trains, duration_us=2000.0)
        assert out.max() <= 1.0
