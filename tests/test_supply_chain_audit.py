# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Supply-chain audit helper tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_tool() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "supply_chain_audit.py"
    spec = importlib.util.spec_from_file_location("supply_chain_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pyproject(path: Path, *, version: str = "1.0.0") -> None:
    path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo-package"',
                f'version = "{version}"',
                'license = "AGPL-3.0-or-later"',
            ]
        ),
        encoding="utf-8",
    )


def _write_sbom(path: Path, *, version: str = "1.0.0") -> None:
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "metadata": {
                    "component": {
                        "bom-ref": "root-component",
                        "name": "demo-package",
                        "version": version,
                        "licenses": [{"license": {"id": "AGPL-3.0-or-later"}}],
                    }
                },
                "components": [{"bom-ref": "dep-1", "name": "numpy", "type": "library"}],
            }
        ),
        encoding="utf-8",
    )


def _write_requirements(path: Path, *, hashed: bool = True) -> None:
    text = [
        "#",
        "#    pip-compile --generate-hashes --output-file=requirements/release.txt requirements/release.in",
        "#",
        "cyclonedx-bom==7.3.0 \\",
    ]
    if hashed:
        text.append("    --hash=sha256:" + "a" * 64)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def test_supply_chain_audit_passes_clean_fixture(tmp_path: Path) -> None:
    tool = _load_tool()
    pyproject = tmp_path / "pyproject.toml"
    sbom = tmp_path / "sbom.cdx.json"
    requirements = tmp_path / "release.txt"
    _write_pyproject(pyproject)
    _write_sbom(sbom)
    _write_requirements(requirements)

    report = tool.audit_supply_chain(
        sbom_path=sbom,
        pyproject_path=pyproject,
        requirements_path=requirements,
    )

    assert report["passed"] is True
    assert report["errors"] == 0
    assert report["warnings"] == 0


def test_supply_chain_audit_warns_on_sbom_version_drift(tmp_path: Path) -> None:
    tool = _load_tool()
    pyproject = tmp_path / "pyproject.toml"
    sbom = tmp_path / "sbom.cdx.json"
    requirements = tmp_path / "release.txt"
    _write_pyproject(pyproject, version="2.0.0")
    _write_sbom(sbom, version="1.0.0")
    _write_requirements(requirements)

    report = tool.audit_supply_chain(
        sbom_path=sbom,
        pyproject_path=pyproject,
        requirements_path=requirements,
    )
    strict_report = tool.audit_supply_chain(
        sbom_path=sbom,
        pyproject_path=pyproject,
        requirements_path=requirements,
        strict=True,
    )

    assert report["passed"] is True
    assert report["warnings"] == 1
    assert strict_report["passed"] is False


def test_supply_chain_audit_fails_unhashed_release_requirement(tmp_path: Path) -> None:
    tool = _load_tool()
    pyproject = tmp_path / "pyproject.toml"
    sbom = tmp_path / "sbom.cdx.json"
    requirements = tmp_path / "release.txt"
    _write_pyproject(pyproject)
    _write_sbom(sbom)
    _write_requirements(requirements, hashed=False)

    report = tool.audit_supply_chain(
        sbom_path=sbom,
        pyproject_path=pyproject,
        requirements_path=requirements,
    )

    assert report["passed"] is False
    assert report["errors"] == 1
