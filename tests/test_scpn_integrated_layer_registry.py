# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerRegistry from former test_scpn_integrated.py

"""Focused suite: TestLayerRegistry from former test_scpn_integrated.py."""

from __future__ import annotations

from sc_neurocore.scpn.layers import (
    LAYER_REGISTRY,
)


class TestLayerRegistry:
    """The registry of SCPN layers."""

    def test_has_16_layers(self) -> None:
        """The registry has 16 layers."""
        assert len(LAYER_REGISTRY) == 16

    def test_keys_l1_to_l16(self) -> None:
        """The registry keys run from l1 to l16."""
        for i in range(1, 17):
            assert f"l{i}" in LAYER_REGISTRY, f"l{i} missing from registry"
