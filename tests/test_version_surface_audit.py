# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations

import tools.version_surface_audit as audit


def test_source_version_surfaces_match_project_version() -> None:
    assert audit.audit_versions(include_installed=False) == []


def test_installed_distribution_metadata_matches_source_tree() -> None:
    failures = audit.audit_versions(include_installed=True)
    assert failures == []


def test_cli_reports_source_only_success(capsys) -> None:
    assert audit.main(["--source-only"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_reports_installed_metadata_mismatch(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        audit,
        "installed_surfaces",
        lambda: [audit.VersionSurface("installed:sc-neurocore", "0.0.0")],
    )

    assert audit.main([]) == 1
    captured = capsys.readouterr()
    assert "installed:sc-neurocore has version 0.0.0" in captured.out
    assert "Refresh the environment" in captured.out


def test_installed_surface_audit_reports_duplicate_stale_dist_info(monkeypatch) -> None:
    monkeypatch.setattr(
        audit,
        "source_surfaces",
        lambda root=audit.ROOT: [audit.VersionSurface("pyproject.toml", "3.15.25")],
    )
    monkeypatch.setattr(
        audit,
        "installed_surfaces",
        lambda: [
            audit.VersionSurface(
                "installed:sc-neurocore:sc_neurocore-3.15.25.dist-info", "3.15.25"
            ),
            audit.VersionSurface("installed:sc-neurocore:sc_neurocore-3.15.0.dist-info", "3.15.0"),
        ],
    )

    assert audit.audit_versions(include_installed=True) == [
        "installed:sc-neurocore:sc_neurocore-3.15.0.dist-info has version 3.15.0, expected 3.15.25"
    ]
