# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExportVisualization from former test_photonic_noc.py

"""Focused suite: TestExportVisualization from former test_photonic_noc.py."""

from __future__ import annotations

import json
import os
from sc_neurocore.bridges.photonic_noc import (
    export_photonic_json,
    PhotonicCircuitDesign,
    visualize_photonic,
)


class TestExportVisualization:
    """Export and visualization tests."""

    def test_export_json(self, simple_design: PhotonicCircuitDesign, tmp_path: str) -> None:
        """The JSON export carries the nodes and waveguides."""
        path = os.path.join(tmp_path, "photonic.json")
        export_photonic_json(simple_design, path)
        with open(path) as f:
            data = json.load(f)
        assert data["n_nodes"] == 4
        assert len(data["waveguides"]) > 0

    def test_visualize(self, simple_design: PhotonicCircuitDesign) -> None:
        """The text view lists waveguides and MZI gates."""
        viz = visualize_photonic(simple_design)
        assert "Photonic NoC" in viz
        assert "Waveguides" in viz
        assert "MZI Gates" in viz
