# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Conda report corruption refusal contracts

"""Validate actual native reports and refuse altered package coverage or scanner policy."""

from __future__ import annotations

import copy

import yaml

import pytest

from tests.test_tools.test_pixi_audit_cli import (
    LOCK,
    ROOT,
    NativeCondaPacket,
    native_pixi_audit as native_pixi_audit,
)
from tools.security_scan.conda_dependency_reports import (
    conda_lock_artifacts,
    grype_matches,
    syft_lock_inventory,
)


@pytest.mark.parametrize(
    "change",
    [
        "omit",
        "repeat",
        "url",
        "digest",
        "name",
        "version",
        "build",
        "cpe",
        "scanner",
        "source",
        "cataloger",
        "metadata",
        "identifier",
    ],
)
def test_native_syft_inventory_refuses_changed_artifact_custody(
    native_pixi_audit: NativeCondaPacket, change: str
) -> None:
    """Reject corrupted copies of the actual locked-environment SBOM.

    Parameters
    ----------
    native_pixi_audit : tuple
        Actual scanner packets from the complete public audit.
    change : str
        Inventory or producer identity to corrupt.
    """
    syft, _, _, prefix, _ = native_pixi_audit
    candidate = copy.deepcopy(syft)
    row = candidate["artifacts"][0]
    if change == "omit":
        candidate["artifacts"].pop()
    elif change == "repeat":
        candidate["artifacts"][1] = copy.deepcopy(row)
    elif change == "url":
        row["metadata"]["url"] += "-changed"
    elif change == "digest":
        row["metadata"]["sha256"] = "0" * 64
    elif change in {"name", "version", "build"}:
        row["metadata"][change] += "-changed"
    elif change == "cpe":
        row["cpes"] = []
    elif change == "scanner":
        candidate["descriptor"]["version"] = "unqualified"
    elif change == "source":
        candidate["source"]["metadata"]["path"] += "-changed"
    elif change == "cataloger":
        row["foundBy"] = "unqualified"
    elif change == "metadata":
        candidate["source"]["metadata"] = None
    elif change == "identifier":
        row["id"] = []
    with pytest.raises(ValueError):
        syft_lock_inventory(candidate, (ROOT / LOCK).read_bytes(), prefix)


@pytest.mark.parametrize(
    "change",
    [
        "matches",
        "ignored",
        "filter",
        "cpe",
        "database",
        "freshness",
        "origin",
        "timestamp",
        "source",
        "scanner",
        "artifact",
        "identity",
        "built",
        "object",
        "stock",
    ],
)
def test_native_grype_report_refuses_filtered_or_unbound_results(
    native_pixi_audit: NativeCondaPacket, change: str
) -> None:
    """Reject corrupted native policy, database and finding bindings.

    Parameters
    ----------
    native_pixi_audit : tuple
        Real Syft and Grype output for the committed lock.
    change : str
        Required scanner setting or input identity to invalidate.
    """
    syft, grype, _, prefix, _ = native_pixi_audit
    artifacts = syft_lock_inventory(syft, (ROOT / LOCK).read_bytes(), prefix)
    candidate = copy.deepcopy(grype)
    descriptor = candidate["descriptor"]
    config = descriptor["configuration"]
    if change == "matches":
        del candidate["matches"]
    elif change == "ignored":
        candidate["ignoredMatches"] = copy.deepcopy(candidate["matches"])
    elif change == "filter":
        config["only-fixed"] = True
    elif change == "cpe":
        config["match"]["stock"]["using-cpes"] = False
    elif change == "database":
        descriptor["db"]["status"]["valid"] = False
    elif change == "freshness":
        config["db"]["require-update-check"] = False
    elif change == "origin":
        config["db"]["update-url"] = "https://example.invalid"
    elif change == "timestamp":
        descriptor["db"]["status"]["built"] = "2020-01-01T00:00:00Z"
    elif change == "source":
        candidate["source"]["target"] += "-changed"
    elif change == "scanner":
        descriptor["version"] = "unqualified"
    elif change == "artifact":
        candidate["matches"][0]["artifact"]["id"] = "unknown"
    elif change == "identity":
        candidate["matches"][0]["artifact"]["version"] += "-changed"
    elif change == "built":
        del descriptor["db"]["status"]["built"]
    elif change == "object":
        candidate["matches"][0]["artifact"] = None
    elif change == "stock":
        config["match"]["stock"] = None
    with pytest.raises(ValueError):
        grype_matches(candidate, artifacts, prefix)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "environments",
        "default",
        "platform",
        "pypi",
        "digest",
        "archive",
        "filename",
        "selection",
        "repeat",
        "empty",
    ],
)
def test_committed_conda_lock_requires_exact_supported_artifact_selection(change: str) -> None:
    """Refuse corrupted copies of the committed lock before scanner execution.

    Parameters
    ----------
    change : str
        Artifact or environment selection to make incomplete.
    """
    payload = yaml.safe_load((ROOT / LOCK).read_bytes())
    if change == "missing":
        del payload["packages"]
    elif change == "environments":
        payload["environments"]["other"] = copy.deepcopy(payload["environments"]["default"])
    elif change == "default":
        payload["environments"]["default"] = None
    elif change == "platform":
        payload["environments"]["default"]["packages"]["win-64"] = []
    elif change == "pypi":
        payload["packages"][0] = {"pypi": "https://example.invalid/wheel"}
    elif change == "digest":
        payload["packages"][0]["sha256"] = "unavailable"
    elif change == "archive":
        payload["packages"][0]["conda"] += ".zip"
    elif change == "filename":
        payload["packages"][0]["conda"] = "https://example.invalid/package.conda"
    elif change == "selection":
        payload["environments"]["default"]["packages"]["linux-64"][0] = {}
    elif change == "repeat":
        payload["environments"]["default"]["packages"]["linux-64"].append(
            copy.deepcopy(payload["environments"]["default"]["packages"]["linux-64"][0])
        )
    elif change == "empty":
        payload["packages"].clear()
    with pytest.raises(ValueError):
        conda_lock_artifacts(yaml.safe_dump(payload).encode())
