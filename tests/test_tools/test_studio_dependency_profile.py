# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio dependency profile metadata

"""Install-profile contract tests for the Studio backend dependencies."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pragma: no cover


REPO_ROOT = Path(__file__).resolve().parents[2]
STARLETTE_TESTCLIENT_TRANSPORT = "httpx2>=2.5,<3"


def _optional_dependencies() -> dict[str, list[str]]:
    """Load optional dependency groups from the project metadata."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, Any], pyproject["project"])
    return cast(dict[str, list[str]], project["optional-dependencies"])


def test_studio_extra_declares_non_deprecated_testclient_transport() -> None:
    """The Studio extra installs Starlette's non-deprecated TestClient transport."""
    optional_dependencies = _optional_dependencies()

    assert STARLETTE_TESTCLIENT_TRANSPORT in optional_dependencies["studio"]
    assert STARLETTE_TESTCLIENT_TRANSPORT in optional_dependencies["full"]


def test_install_profile_docs_list_studio_testclient_transport() -> None:
    """Install-profile docs name both Studio HTTP client transports."""
    docs = (REPO_ROOT / "docs" / "guides" / "install_profiles.md").read_text(encoding="utf-8")

    assert (
        '| `pip install "sc-neurocore[studio]"` | Web studio / local design UI | '
        "`fastapi`, `uvicorn`, `httpx`, `httpx2`; `nir` for the canvas's NIR export and import; "
        "`pint` and `sympy` for candidate-model units and equation diffs |"
    ) in docs


def test_studio_extra_installs_what_the_canvas_nir_export_imports() -> None:
    """The canvas offers NIR export and import, so the Studio extra installs nir.

    It carries the same pin as the ``nir`` extra: the NIR bridge and the
    Studio read and write the same files.
    """
    optional_dependencies = _optional_dependencies()

    assert optional_dependencies["nir"] == ["nir>=1.0,<1.0.9"]
    assert "nir>=1.0,<1.0.9" in optional_dependencies["studio"]


def test_studio_extra_installs_what_candidate_packages_import() -> None:
    """Candidate units are parsed with pint and equations diffed with SymPy.

    The Studio extra carries the pins the ``hdl`` and ``symbolic`` extras and the
    full profile already use, so every profile resolves the same libraries.
    """
    optional_dependencies = _optional_dependencies()

    for requirement, extra in (("pint>=0.23", "hdl"), ("sympy>=1.12", "symbolic")):
        assert requirement in optional_dependencies[extra]
        assert requirement in optional_dependencies["studio"]
        assert requirement in optional_dependencies["full"]
