# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sphinx documentation build contracts

"""Keep the secondary Sphinx API site reproducible and version-current."""

import runpy
from pathlib import Path

import yaml

from sc_neurocore import __version__
from tests.public_api_support import _project_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_sphinx_config_uses_canonical_version_without_missing_static_path() -> None:
    """The API site must derive its release and avoid nonexistent static input."""
    config = runpy.run_path(str(ROOT / "docs/sphinx/source/conf.py"))

    assert config["release"] == __version__
    assert config["version"] == ".".join(__version__.split(".")[:2])
    assert config["html_theme"] == "furo"
    assert config["html_static_path"] == []


def test_docs_profiles_declare_sphinx_and_theme() -> None:
    """Both public metadata and the CI lock input must own Sphinx dependencies."""
    docs_extra = _project_metadata()["project"]["optional-dependencies"]["docs"]
    docs_input = (ROOT / "requirements/docs.in").read_text(encoding="utf-8").lower()

    assert any(requirement.startswith("sphinx") for requirement in docs_extra)
    assert any(requirement.startswith("furo") for requirement in docs_extra)
    assert "sphinx" in docs_input
    assert "furo" in docs_input


def test_docs_workflow_builds_sphinx_with_warnings_fatal() -> None:
    """CI must execute the committed API-doc build without masking warnings."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build-docs"]["steps"]
    command = next(
        step["run"] for step in steps if step.get("name") == "Build Sphinx API documentation"
    )

    assert "sphinx-build -W --keep-going" in command
    assert "docs/sphinx/source" in command
