# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerSpec from former test_extract_shd_weights.py

"""Focused suite: TestLayerSpec from former test_extract_shd_weights.py."""

from __future__ import annotations

from extract_shd_weights_support import (
    SHD_LAYERS,
)


class TestLayerSpec:
    """Pin the three-layer SHD network the extraction expects."""

    def test_three_layers_defined(self) -> None:
        """The SHD network has exactly three weight layers."""
        assert len(SHD_LAYERS) == 3

    def test_layer_dimensions_match_architecture(self) -> None:
        """The layers map 140 inputs through 128 and 128 to 20 outputs."""
        # 140 → 128 → 128 → 20
        assert SHD_LAYERS[0].in_features == 140
        assert SHD_LAYERS[0].out_features == 128
        assert SHD_LAYERS[1].in_features == 128
        assert SHD_LAYERS[1].out_features == 128
        assert SHD_LAYERS[2].in_features == 128
        assert SHD_LAYERS[2].out_features == 20

    def test_only_first_two_layers_have_delays(self) -> None:
        """Only the two hidden layers carry axonal delays."""
        assert SHD_LAYERS[0].delay_key is not None
        assert SHD_LAYERS[1].delay_key is not None
        assert SHD_LAYERS[2].delay_key is None  # output layer has no axonal delay
