# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic co-design compile evidence

"""Successful compile and export evidence for stochastic photonic co-design."""

import json

import numpy as np

from sc_neurocore.bridges import PhotonicCoDesignConfig, StochasticPhotonicCoDesignLoop
from sc_neurocore.optics.photonic_emitter import PhotonicTarget


def test_codesign_loop_compiles_full_evidence_surface(tmp_path) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    adjacency = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 0.25],
            [0.75, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    config = PhotonicCoDesignConfig(
        bitstream_length=512,
        seed=0x2222,
        run_fdtd=True,
        fdtd_steps=4,
        target=PhotonicTarget.lightmatter(),
    )

    report = StochasticPhotonicCoDesignLoop(config).compile(
        adjacency,
        probabilities=[0.35, 0.55, 0.75],
        node_labels=["sensor", "hidden", "actuator"],
        name="photonic_loop_test",
    )

    assert report.name == "photonic_loop_test"
    assert report.design.n_nodes == 3
    assert len(report.bitstreams) == 3
    assert len(report.optical_results) == 3
    assert report.fdtd["enabled"] is True
    assert report.fdtd["energy"] > 0.0
    assert report.layout_manifest["gdsii_status"] == "handoff_manifest_only"
    assert report.power_budget["n_paths"] > 0
    assert report.crosstalk["n_channels"] == 3
    assert len(report.scc_matrix) == 3

    out_path = tmp_path / "photonic_report.json"
    report.export_json(out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["name"] == "photonic_loop_test"
    assert payload["design"]["n_wdm_channels"] == 3
