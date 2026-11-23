# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiModalFusionAttention from former test_fusion_multimodal.py

"""Focused suite: TestMultiModalFusionAttention from former test_fusion_multimodal.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.fusion.multimodal import (
    ModalityConfig,
    MultiModalFusion,
)


class TestMultiModalFusionAttention:
    """Attention-weighted fusion."""

    def test_attention_mode(self) -> None:
        """Attention weights cover each modality and sum to 1."""
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
