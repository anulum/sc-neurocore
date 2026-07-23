# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtractLayers from former test_conversion.py

"""Focused suite: TestExtractLayers from former test_conversion.py."""

from __future__ import annotations

from tests.conversion_support import *  # noqa: F403

class TestExtractLayers:
    def test_extracts_linear(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layers = _extract_layers(model)
        assert len(layers) == 2
        assert layers[0][0].shape == (8, 4)
        assert layers[1][0].shape == (3, 8)

    def test_handles_no_bias(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8, bias=False))
        layers = _extract_layers(model)
        assert layers[0][1] is None
