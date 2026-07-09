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
        "`fastapi`, `uvicorn`, `httpx`, `httpx2` |"
    ) in docs
