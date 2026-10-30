# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Release metadata states one version everywhere

"""Bind the version that archives and citations take from the tagged tree to the package version."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_archive_citation_and_packages_state_the_same_version() -> None:
    """A tag cannot carry a stale archive label.

    The research archive labels a release with the version written in
    ``.zenodo.json`` of the tagged tree, and citation tools read
    ``CITATION.cff``. Both must equal the version of the Python package and of
    the engine built from the same tree; a release whose archive record is
    labelled with an older version misleads everyone who cites by version.
    The conda recipe and the root component of the committed SBOM name the
    same release.
    """
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    engine = tomllib.loads((ROOT / "bridge/pyproject.toml").read_text(encoding="utf-8"))["project"]
    crate = tomllib.loads((ROOT / "engine/Cargo.toml").read_text(encoding="utf-8"))["package"]
    archive = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    recipe = re.findall(
        r'^\{% set version = "([^"]+)" %\}$',
        (ROOT / "conda/meta.yaml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))["metadata"]["component"]
    assert len(recipe) == 1, "The conda recipe sets its version exactly once."
    assert sbom["name"] == package["name"], "The SBOM root component is the Python package."
    versions = {
        "pyproject.toml": package["version"],
        "bridge/pyproject.toml": engine["version"],
        "engine/Cargo.toml": crate["version"],
        ".zenodo.json": archive["version"],
        "CITATION.cff": str(citation["version"]),
        "conda/meta.yaml": recipe[0],
        "sbom.cdx.json": sbom["version"],
    }
    assert len(set(versions.values())) == 1, versions
    assert not str(archive["version"]).startswith("v"), "The archive label carries no prefix."


def test_committed_sbom_lists_exactly_the_declared_base_install_contract() -> None:
    """The root SBOM names every base dependency with its declared range, and nothing else.

    The committed ``sbom.cdx.json`` describes the install contract in
    ``pyproject.toml``; resolved versions belong to the environment SBOMs that
    the release attaches. A root SBOM generated from an older requirements
    file once listed two of the five dependencies with ranges the package no
    longer declares, while the offline audit passed because it checks only the
    root component.
    """
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    sbom = json.loads((ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
    listed = sorted(
        component["description"].split(": ", maxsplit=1)[1] for component in sbom["components"]
    )

    assert listed == sorted(declared), "Regenerate sbom.cdx.json from pyproject.toml."
    assert {entry["ref"] for entry in sbom["dependencies"]} == {
        "root-component",
        *(component["bom-ref"] for component in sbom["components"]),
    }
