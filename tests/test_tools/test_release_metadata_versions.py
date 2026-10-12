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
    """
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    engine = tomllib.loads((ROOT / "bridge/pyproject.toml").read_text(encoding="utf-8"))["project"]
    crate = tomllib.loads((ROOT / "engine/Cargo.toml").read_text(encoding="utf-8"))["package"]
    archive = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    versions = {
        "pyproject.toml": package["version"],
        "bridge/pyproject.toml": engine["version"],
        "engine/Cargo.toml": crate["version"],
        ".zenodo.json": archive["version"],
        "CITATION.cff": str(citation["version"]),
    }
    assert len(set(versions.values())) == 1, versions
    assert not str(archive["version"]).startswith("v"), "The archive label carries no prefix."
