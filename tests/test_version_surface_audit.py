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
import sys

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
        "src/sc_neurocore/_version.py": f'VERSION: Final[str] = "{version}"\n',
        "src/sc_neurocore/__init__.py": f"{audit.INIT_IMPORT}\n\n{audit.INIT_REEXPORT}\n",
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
        "src/sc_neurocore/_version.py",
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


def test_a_version_file_without_a_version_is_refused_by_its_label(tmp_path: Path) -> None:
    """A checkout whose version file defines no ``VERSION`` is named, wherever the checkout is.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    """
    _version_tree(tmp_path, src_sc_neurocore__version_py='"""No version here."""\n')
    with pytest.raises(
        RuntimeError, match=re.escape("src/sc_neurocore/_version.py does not define VERSION")
    ):
        audit.audit_versions(include_installed=False, root=tmp_path)


@pytest.mark.parametrize(
    ("init_text", "message"),
    [
        ('__version__ = "9.9.9"\n', "states its own __version__; re-export VERSION"),
        ('__version__: str = "9.9.9"\n', "states its own __version__; re-export VERSION"),
        (
            "from sc_neurocore import _version\n"
            "__version__: str = _version.VERSION\n"
            '__version__ = "9.9.9"\n',
            "states its own __version__; re-export VERSION",
        ),
        (
            "from sc_neurocore._version import VERSION\n__version__ = VERSION\n",
            "states its own __version__; re-export VERSION",
        ),
        ('"""No re-export."""\n', "does not re-export VERSION as __version__"),
        ("__version__: str = _version.VERSION\n", "does not re-export VERSION as __version__"),
    ],
)
def test_a_package_root_that_does_not_re_export_the_version_is_refused(
    tmp_path: Path, init_text: str, message: str
) -> None:
    """The package root may only re-export ``VERSION``; any version of its own is refused.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    init_text : str
        The package root's text.
    message : str
        The refusal it must produce.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=init_text)
    with pytest.raises(RuntimeError, match=re.escape(f"src/sc_neurocore/__init__.py {message}")):
        audit.audit_versions(include_installed=False, root=tmp_path)


#: The correct re-export, after which each rebinding below is added.
_REEXPORT = f"{audit.INIT_IMPORT}\n\n{audit.INIT_REEXPORT}\n"


@pytest.mark.parametrize(
    "rebinding",
    [
        'if True:\n    __version__ = "9.9.9"\n',
        'globals()["__version__"] = "9.9.9"\n',
        'import sys\nsetattr(sys.modules[__name__], "__version__", "9.9.9")\n',
        '__version__ += "-local"\n',
        'if (__version__ := "9.9.9"):\n    pass\n',
        "for __version__ in ['9.9.9']:\n    pass\n",
        "_ = [__version__ for __version__ in ['9.9.9']]\n",
        "import contextlib\nwith contextlib.nullcontext('9.9.9') as __version__:\n    pass\n",
        "try:\n    pass\nexcept ValueError as __version__:\n    pass\n",
        "del __version__\n",
        "from os import sep as __version__\n",
        "def f() -> None:\n    global __version__\n",
        "def __version__() -> str:\n    return '9.9.9'\n",
        "class __version__:\n    pass\n",
        "a, (b, *__version__) = 1, (2, 3)\n",
        'match "9.9.9":\n    case __version__:\n        pass\n',
        'match "9.9.9":\n    case str() as __version__:\n        pass\n',
        "match ['9.9.9']:\n    case [*__version__]:\n        pass\n",
        "match {'v': '9.9.9'}:\n    case {**__version__}:\n        pass\n",
    ],
)
def test_every_other_binding_of_the_version_name_is_refused(tmp_path: Path, rebinding: str) -> None:
    """A rebinding at any depth, by any form, or through a string is a version of the root's own.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    rebinding : str
        A second binding of ``__version__``, added after the correct re-export.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=_REEXPORT + rebinding)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "src/sc_neurocore/__init__.py states its own __version__; re-export VERSION"
        ),
    ):
        audit.audit_versions(include_installed=False, root=tmp_path)


def test_a_type_alias_of_the_version_name_is_refused(tmp_path: Path) -> None:
    """``type __version__ = str`` binds the name from Python 3.12, and cannot even parse before it.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=_REEXPORT + "type __version__ = str\n")
    message = (
        "states its own __version__; re-export VERSION"
        if sys.version_info >= (3, 12)
        else "cannot be parsed"
    )
    with pytest.raises(RuntimeError, match=re.escape(f"src/sc_neurocore/__init__.py {message}")):
        audit.audit_versions(include_installed=False, root=tmp_path)


@pytest.mark.parametrize(
    "harmless",
    [
        '__all__ = ["__version__"]\n',
        "__all__: list[str] = ['__version__']\n",
        '__all__ += ["__version__"]\n',
        "obj.__version__ = 1\n",
        'values = {"key": "__version__x"}\n',
    ],
)
def test_the_export_list_and_attribute_targets_are_not_bindings(
    tmp_path: Path, harmless: str
) -> None:
    """Naming ``__version__`` in ``__all__``, or setting another object's attribute, is allowed.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    harmless : str
        A statement that does not bind the root's ``__version__``.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=_REEXPORT + harmless)
    assert audit.audit_versions(include_installed=False, root=tmp_path) == []


@pytest.mark.parametrize(
    "init_text",
    [
        "import sc_neurocore._version as _version\n" + audit.INIT_REEXPORT + "\n",
        "from sc_neurocore import _version as v\n" + audit.INIT_REEXPORT + "\n",
        "from . import _version\n" + audit.INIT_REEXPORT + "\n",
    ],
)
def test_the_version_module_must_be_imported_by_the_selected_line(
    tmp_path: Path, init_text: str
) -> None:
    """Only ``from sc_neurocore import _version`` counts as importing the version module.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    init_text : str
        A root that imports the module by another form.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=init_text)
    with pytest.raises(
        RuntimeError,
        match=re.escape("src/sc_neurocore/__init__.py does not re-export VERSION as __version__"),
    ):
        audit.audit_versions(include_installed=False, root=tmp_path)


def test_a_package_root_that_cannot_be_parsed_is_refused(tmp_path: Path) -> None:
    """A root with a syntax error is refused by its label and the parser's reason.

    Parameters
    ----------
    tmp_path : Path
        A checkout outside this repository.
    """
    _version_tree(tmp_path, src_sc_neurocore___init___py=_REEXPORT + "def (:\n")
    with pytest.raises(
        RuntimeError, match=re.escape("src/sc_neurocore/__init__.py cannot be parsed")
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
