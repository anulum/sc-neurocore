# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Photonic Co-Design Tests

from __future__ import annotations

import json

import numpy as np
import pytest

from sc_neurocore.bridges import (
    PhotonicCoDesignConfig,
    StochasticPhotonicCoDesignLoop,
    derive_probabilities_from_adjacency,
    encode_bitstream_bank,
)
from sc_neurocore.optics.photonic_emitter import PhotonicTarget


def test_derive_probabilities_from_adjacency_uses_inbound_weight_mass() -> None:
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    probabilities = derive_probabilities_from_adjacency(adjacency)

    np.testing.assert_allclose(
        probabilities,
        np.array([1.0 / 65535.0, 1.0 - 1.0 / 65535.0, 1.0 / 3.0]),
    )


def test_encode_bitstream_bank_is_deterministic_and_density_bounded() -> None:
    first = encode_bitstream_bank(
        [0.25, 0.75],
        bitstream_length=512,
        seed=0x1234,
        names=["low", "high"],
    )
    second = encode_bitstream_bank(
        [0.25, 0.75],
        bitstream_length=512,
        seed=0x1234,
        names=["low", "high"],
    )

    assert first == second
    assert first[0].name == "low"
    assert first[1].measured_probability > first[0].measured_probability
    assert first[0].density_error < 0.08
    assert first[1].transitions > 0


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        ([[0.5]], "one-dimensional"),
        ([-0.1], r"\[0, 1\]"),
        ([1.1], r"\[0, 1\]"),
    ],
)
def test_encode_bitstream_bank_rejects_invalid_probabilities(
    probabilities: list[float] | list[list[float]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        encode_bitstream_bank(probabilities, bitstream_length=128, seed=1)


def test_codesign_loop_compiles_full_evidence_surface(tmp_path) -> None:
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


def test_codesign_loop_reports_blockers_without_hiding_bad_margins() -> None:
    adjacency = np.ones((4, 4), dtype=np.float64) - np.eye(4, dtype=np.float64)
    config = PhotonicCoDesignConfig(
        bitstream_length=256,
        min_power_margin_db=100.0,
        max_crosstalk_db=-80.0,
        run_fdtd=False,
    )

    report = StochasticPhotonicCoDesignLoop(config).compile(
        adjacency,
        probabilities=[0.2, 0.4, 0.6, 0.8],
    )

    assert report.feasible is False
    assert any("worst optical margin" in blocker for blocker in report.blockers)
    assert any("worst crosstalk" in blocker for blocker in report.blockers)


def test_codesign_loop_rejects_mismatched_inputs() -> None:
    loop = StochasticPhotonicCoDesignLoop(PhotonicCoDesignConfig(run_fdtd=False))
    with pytest.raises(ValueError, match="at least one node"):
        loop.compile(np.zeros((0, 0), dtype=np.float64))
    with pytest.raises(ValueError, match="node_labels"):
        loop.compile(np.eye(2), node_labels=["only_one"])
    with pytest.raises(ValueError, match="probabilities"):
        loop.compile(np.eye(2), probabilities=[0.5, 0.5, 0.5])
