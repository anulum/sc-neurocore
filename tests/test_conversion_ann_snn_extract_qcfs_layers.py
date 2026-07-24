# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtractQcfsLayers from former test_conversion_ann_snn.py

"""Focused suite: TestExtractQcfsLayers from former test_conversion_ann_snn.py."""

from __future__ import annotations

from tests.conversion_ann_snn_support import *  # noqa: F403


class TestExtractQcfsLayers:
    def test_returns_theta_and_t_in_order(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 8),
            QCFSActivation(T=4, theta=2.0),
            nn.Linear(8, 3),
            QCFSActivation(T=4, theta=3.0),
        )
        layers = _extract_qcfs_layers(model)
        assert layers == [(2.0, 4), (3.0, 4)]

    def test_empty_for_relu_model(self) -> None:
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        assert _extract_qcfs_layers(model) == []
