# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Each surface that states the release version is held to the project version, with a remedy."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

import tools.version_surface_audit as audit


def test_source_version_surfaces_match_project_version() -> None:
    """Every checked-in surface of this checkout states the project version."""
    assert audit.audit_versions(include_installed=False) == []


def test_installed_distribution_metadata_matches_source_tree() -> None:
    """The installed distributions state the checkout's version too."""
    failures = audit.audit_versions(include_installed=True)
    assert failures == []


def test_cli_reports_source_only_success(capsys: pytest.CaptureFixture[str]) -> None:
    """A clean source-only run prints nothing and exits 0.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Captures the command's output.
    """
    assert audit.main(["--source-only"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_reports_installed_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale installed distribution is named, with the reinstall as its only remedy.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the installed distributions with a stale one.
    capsys : pytest.CaptureFixture[str]
        Captures the command's output.
    """
    monkeypatch.setattr(
        audit,
        "installed_surfaces",
        lambda: [audit.VersionSurface("installed:sc-neurocore", "0.0.0")],
    )

    assert audit.main([]) == 1
    captured = capsys.readouterr()
    assert "installed:sc-neurocore has version 0.0.0" in captured.out
    assert "Refresh the environment" in captured.out
    assert "Edit each named file" not in captured.out


def test_installed_surface_audit_reports_duplicate_stale_dist_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale ``.dist-info`` left beside the current one is named by its folder.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the source and installed surfaces.
    """
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


#: The release metadata files the audit reads besides the package and engine manifests.
METADATA_SURFACES = (".zenodo.json", "CITATION.cff", "conda/meta.yaml")


def _version_tree(root: Path, version: str = "9.9.9", **overrides: str) -> None:
    """Write every audited version surface under ``root``, all at ``version`` unless overridden.

    Parameters
    ----------
    root : Path
        The tree to write.
    version : str
        The version every surface states.
    **overrides : str
        A file's text by its label with ``/`` and ``.`` as ``_``, in place of the default.
    """
    texts = {
        "pyproject.toml": f'[project]\nname = "sc-neurocore"\nversion = "{version}"\n',
        "src/sc_neurocore/__init__.py": f'__version__ = "{version}"\n',
        "engine/Cargo.toml": f'[package]\nname = "sc-neurocore-engine"\nversion = "{version}"\n',
        "bridge/pyproject.toml": f'[project]\nname = "sc-neurocore-bridge"\nversion = "{version}"\n',
        ".zenodo.json": f'{{"title": "SC-NeuroCore", "version": "{version}"}}\n',
        "CITATION.cff": f"cff-version: 1.2.0\ntitle: SC-NeuroCore\nversion: {version}\n",
        "conda/meta.yaml": f'{{% set version = "{version}" %}}\npackage:\n  version: {{{{ version }}}}\n',
    }
    for label, text in texts.items():
        key = label.replace("/", "_").replace(".", "_")
        path = root / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(overrides.get(key, text), encoding="utf-8")


def test_release_metadata_files_are_audited() -> None:
    """Zenodo, citation and conda metadata state the release version and are held to it."""
    labels = [surface.label for surface in audit.source_surfaces()]
    assert labels[:4] == [
        "pyproject.toml",
        "src/sc_neurocore/__init__.py",
        "engine/Cargo.toml",
        "bridge/pyproject.toml",
    ]
    assert set(METADATA_SURFACES) <= set(labels)


@pytest.mark.parametrize(
    ("label", "stale"),
    [
        (".zenodo.json", '{"title": "SC-NeuroCore", "version": "9.9.8"}\n'),
        ("CITATION.cff", "cff-version: 1.2.0\nversion: '9.9.8'\n"),
        ("conda/meta.yaml", '{% set version = "9.9.8" %}\n'),
    ],
)
def test_a_stale_release_metadata_version_is_reported(
    tmp_path: Path, label: str, stale: str
) -> None:
    """A metadata file left at an earlier version is named against the project version.

    Parameters
    ----------
    tmp_path : Path
        The tree.
    label : str
        The stale file.
    stale : str
        Its text, at version 9.9.8.
    """
    _version_tree(tmp_path, **{label.replace("/", "_").replace(".", "_"): stale})
    assert audit.audit_versions(include_installed=False, root=tmp_path) == [
        f"{label} has version 9.9.8, expected 9.9.9"
    ]


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (".zenodo.json", '{"title": "SC-NeuroCore"}\n'),
        (".zenodo.json", "[]\n"),
        ("CITATION.cff", "cff-version: 1.2.0\ntitle: SC-NeuroCore\n"),
        ("conda/meta.yaml", "package:\n  name: sc-neurocore\n"),
    ],
)
def test_release_metadata_without_a_version_is_refused(
    tmp_path: Path, label: str, text: str
) -> None:
    """A metadata file that states no version is refused by name, never read as a match.

    Parameters
    ----------
    tmp_path : Path
        The tree.
    label : str
        The file without a version.
    text : str
        Its text.
    """
    _version_tree(tmp_path, **{label.replace("/", "_").replace(".", "_"): text})
    with pytest.raises(RuntimeError, match=re.escape(f"{label} does not state a version")):
        audit.audit_versions(include_installed=False, root=tmp_path)


def test_a_package_file_without_a_version_is_refused_by_its_label(tmp_path: Path) -> None:
    """A checkout whose package file defines no ``__version__`` is named, wherever the checkout is.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py='"""No version here."""\n')
    with pytest.raises(
        RuntimeError, match=re.escape("src/sc_neurocore/__init__.py does not define __version__")
    ):
        audit.audit_versions(include_installed=False, root=tmp_path)


def test_a_stale_checked_in_file_is_told_to_edit_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale release metadata file is not answered with a reinstall, which cannot change it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the source surfaces with one whose ``CITATION.cff`` is left at 9.9.8.
    capsys : pytest.CaptureFixture[str]
        Captures the command's output.
    """
    monkeypatch.setattr(
        audit,
        "source_surfaces",
        lambda root=audit.ROOT: [
            audit.VersionSurface("pyproject.toml", "9.9.9"),
            audit.VersionSurface("CITATION.cff", "9.9.8"),
        ],
    )
    assert audit.main(["--source-only"]) == 1
    out = capsys.readouterr().out
    assert "- CITATION.cff has version 9.9.8, expected 9.9.9" in out
    assert "Edit each named file to state the project version (pyproject.toml)" in out
    assert "Refresh the environment" not in out


def test_both_kinds_of_stale_surface_get_both_remedies() -> None:
    """A stale file and a stale installation each get their own remedy, the file's first."""
    assert audit.remedies(
        [
            "CITATION.cff has version 9.9.8, expected 9.9.9",
            "installed:sc-neurocore has version 9.9.8, expected 9.9.9",
        ]
    ) == [
        "Edit each named file to state the project version (pyproject.toml); "
        "these files are not generated.",
        "Refresh the environment with: python -m pip install -e .",
    ]
    assert audit.remedies([]) == []
