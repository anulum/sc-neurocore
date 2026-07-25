# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN bridge artifact-model contracts

from __future__ import annotations


import numpy as np
import pytest

from scpn_neurocore.bridge import QPUBridgeArtifact


def test_qpu_artifact_hash_rejects_non_finite_json_metadata() -> None:
    artifact = QPUBridgeArtifact(
        domain="power_grid",
        source_name="unit-grid",
        source_mode="fixture",
        K_nm=np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        omega=np.ones(2, dtype=np.float64),
        theta0=np.zeros(2, dtype=np.float64),
        layer_assignments=[0, 1],
        normalization="unit",
        extraction_method="unit_fixture",
        replay_id="fixture:unit-grid:n2",
        metadata={"bad": float("nan")},
    )

    with pytest.raises(ValueError, match="strict finite JSON"):
        artifact.to_qpu_artifact_dict()


def test_qpu_artifact_rejects_empty_identity_and_provenance_fields() -> None:
    base = {
        "domain": "power_grid",
        "source_name": "unit-grid",
        "source_mode": "fixture",
        "K_nm": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        "omega": np.ones(2, dtype=np.float64),
        "theta0": np.zeros(2, dtype=np.float64),
        "layer_assignments": [0, 1],
        "normalization": "unit",
        "extraction_method": "unit_fixture",
        "replay_id": "fixture:unit-grid:n2",
    }
    for key in ("domain", "source_name", "normalization", "extraction_method"):
        bad = dict(base)
        bad[key] = ""

        with pytest.raises(ValueError, match=key):
            QPUBridgeArtifact(**bad)


def test_qpu_artifact_rejects_invalid_layer_assignments() -> None:
    base = {
        "domain": "power_grid",
        "source_name": "unit-grid",
        "source_mode": "fixture",
        "K_nm": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
        "omega": np.ones(2, dtype=np.float64),
        "theta0": np.zeros(2, dtype=np.float64),
        "normalization": "unit",
        "extraction_method": "unit_fixture",
        "replay_id": "fixture:unit-grid:n2",
    }
    bad_assignments = ([0, 0], [0, -1], [0, 1.5], [True, 1])

    for layer_assignments in bad_assignments:
        with pytest.raises(ValueError, match="layer_assignments"):
            QPUBridgeArtifact(**base, layer_assignments=layer_assignments)
