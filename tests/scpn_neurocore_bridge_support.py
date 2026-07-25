# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN quantum-control bridge fixtures

from __future__ import annotations

import hashlib
import json

import numpy as np

import scpn_neurocore.bridge as bridge
from scpn_neurocore.bridge import QPU_ARTIFACT_SCHEMA_VERSION, QPUBridgeArtifact


def validate_qpu_payload(payload: dict[str, object]) -> None:
    validator = getattr(bridge, "validate_qpu_artifact_payload", None)
    assert callable(validator)
    validator(payload)


def refresh_artifact_hash(payload: dict[str, object]) -> dict[str, object]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(body, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def assert_qpu_artifact(artifact: QPUBridgeArtifact, n: int, mode: str) -> None:
    payload = artifact.to_qpu_artifact_dict()
    assert payload["schema_version"] == QPU_ARTIFACT_SCHEMA_VERSION
    assert artifact.source_mode == mode
    assert artifact.K_nm.shape == (n, n)
    assert artifact.omega.shape == (n,)
    assert artifact.theta0 is None or artifact.theta0.shape == (n,)
    assert len(artifact.layer_assignments) == n
    assert np.all(np.isfinite(artifact.K_nm))
    assert np.all(np.isfinite(artifact.omega))
    assert np.allclose(artifact.K_nm, artifact.K_nm.T)
    assert np.allclose(np.diag(artifact.K_nm), 0.0)
    assert np.all(artifact.K_nm >= 0.0)
    assert payload["hashes"]["K_nm_sha256"]
    assert payload["hashes"]["omega_sha256"]
    assert payload["artifact_sha256"]
