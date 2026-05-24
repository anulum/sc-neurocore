# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from pathlib import Path

from sc_neurocore.viz.web_viz import WebVisualizer


class _Layer:
    n_neurons = 7


def test_web_visualizer_writes_topology_html_with_layer_metadata(tmp_path: Path):
    output = tmp_path / "network.html"

    WebVisualizer.generate_html([_Layer(), _Layer()], filename=str(output))

    html = output.read_text()
    assert "SC-NeuroCore Topology" in html
    assert '"id": "Input"' in html
    assert '"id": "L0__Layer"' in html
    assert '"source": "L0__Layer", "target": "L1__Layer"' in html
    assert '"neurons": 7' in html
