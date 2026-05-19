# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    spec = importlib.util.spec_from_file_location("capability_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_scans_core_capability_surfaces() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())

    assert manifest["schema_version"] == tool.CAPABILITY_MANIFEST_SCHEMA_VERSION
    assert manifest["generated_from"]["config"] == "tools/capability_manifest.toml"
    assert manifest["project"]["name"] == "sc-neurocore"
    assert manifest["project"]["readme"] == "README.md"
    assert manifest["counts"]["public_api_exports"] == len(manifest["package_exports"])
    assert manifest["counts"]["python_model_classes"] == len(manifest["models"]["python_classes"])
    assert manifest["counts"]["rust_pyo3_model_wrappers"] == len(
        manifest["models"]["rust_pyo3_wrappers"]
    )
    assert "full" in manifest["packaging"]["optional_extras"]
    assert ".github/workflows/ci.yml" in manifest["quality_gates"]["github_workflows"]
    assert "tests/test_public_api.py" in manifest["quality_gates"]["test_files"]
    assert "adex.md" in manifest["models"]["documentation_pages"]


def test_manifest_validation_rejects_count_drift() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    manifest["counts"]["python_model_classes"] += 1

    report = tool.validate_manifest(manifest)

    assert not report["passed"]
    assert "counts.python_model_classes does not match list length" in report["errors"]


def test_generated_outputs_are_current() -> None:
    tool = _load_tool()

    tool.assert_outputs_current(_repo_root())


def test_readme_snapshot_matches_generated_markdown() -> None:
    tool = _load_tool()
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    start = "<!-- capability-snapshot:start -->"
    end = "<!-- capability-snapshot:end -->"

    block = readme.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0].strip()

    assert (
        block == tool.render_markdown_snapshot(tool.build_capability_manifest(_repo_root())).strip()
    )


def test_markdown_snapshot_is_readme_safe() -> None:
    tool = _load_tool()
    manifest = tool.build_capability_manifest(_repo_root())
    snapshot = tool.render_markdown_snapshot(manifest)

    assert "do not edit counts by hand" in snapshot
    assert f"| Package version | {manifest['project']['version']} |" in snapshot
    assert "Evidence boundary" in snapshot


def test_refresh_outputs_updates_configured_readme_block() -> None:
    tool = _load_tool()
    with _tempdir() as repo:
        _write_portable_fixture(repo)
        config = tool.load_config(repo)

        json_path, markdown_path, readme_path = tool.refresh_outputs(repo, config=config)
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        readme = (repo / "README.md").read_text(encoding="utf-8")

        assert readme_path == repo / "README.md"
        assert markdown_path == repo / "docs/_generated/capability_snapshot.md"
        assert manifest["project_label"] == "Portable Project"
        assert manifest["counts"]["public_api_exports"] == 1
        assert manifest["counts"]["python_model_classes"] == 1
        assert "### Portable Project Capability Inventory" in readme
        assert "| Portable API exports | 1 |" in readme

        tool.assert_outputs_current(repo, config=config)


def test_cli_writes_valid_manifest_and_markdown() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with _tempdir() as tmpdir:
        json_path = tmpdir / "capability_manifest.json"
        markdown_path = tmpdir / "capability_snapshot.md"
        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(_repo_root()),
                "--output",
                str(json_path),
                "--markdown-output",
                str(markdown_path),
                "--no-readme",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "capability-manifest.v1"
        assert markdown_path.read_text(encoding="utf-8").startswith("<!-- SPDX-License-Identifier")

        validate = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--validate",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert validate.returncode == 0


def test_cli_uses_portable_config_and_refreshes_readme() -> None:
    tool_path = _repo_root() / "tools" / "capability_manifest.py"
    with _tempdir() as repo:
        _write_portable_fixture(repo)

        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--repo",
                str(repo),
                "--config",
                "tools/capability_manifest.toml",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "Refreshed" in result.stdout
        assert (repo / "docs/_generated/capability_manifest.json").exists()
        assert "Portable Project Capability Inventory" in (repo / "README.md").read_text(
            encoding="utf-8"
        )


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def _write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_portable_fixture(repo: Path) -> None:
    _write_file(
        repo / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'name = "portable-project"',
                'version = "1.2.3"',
                'requires-python = ">=3.11"',
                'readme = "README.md"',
                'license = "AGPL-3.0-or-later"',
                "",
                "[project.optional-dependencies]",
                'analysis = ["numpy"]',
                "",
            ]
        ),
    )
    _write_file(
        repo / "README.md",
        "\n".join(
            [
                "# Portable Project",
                "",
                "<!-- capability-snapshot:start -->",
                "stale",
                "<!-- capability-snapshot:end -->",
                "",
            ]
        ),
    )
    _write_file(repo / "src/portable_project/__init__.py", '__all__ = ["PortableModel"]\n')
    _write_file(
        repo / "src/portable_project/models/portable.py",
        "class PortableModel:\n    pass\n",
    )
    _write_file(repo / "docs/api/models/portable.md", "# Portable model\n")
    _write_file(repo / "docs/internal/private.md", "# Private\n")
    _write_file(repo / "tests/test_portable.py", "def test_portable() -> None:\n    assert True\n")
    _write_file(repo / ".github/workflows/ci.yml", "name: CI\non: [push]\njobs: {}\n")
    _write_file(
        repo / "tools/capability_manifest.toml",
        "\n".join(
            [
                'project_label = "Portable Project"',
                'schema_version = "capability-manifest.v1"',
                'exclude_doc_parts = ["internal", "_generated"]',
                "",
                "[paths]",
                'json_output = "docs/_generated/capability_manifest.json"',
                'markdown_output = "docs/_generated/capability_snapshot.md"',
                'package_root = "src/portable_project"',
                'model_sources = "src/portable_project/models"',
                'model_docs = "docs/api/models"',
                'tests_root = "tests"',
                'docs_root = "docs"',
                'workflows_root = ".github/workflows"',
                'rust_wrappers = "engine/src/pyo3_neurons.rs"',
                "",
                "[readme]",
                'path = "README.md"',
                'marker_start = "<!-- capability-snapshot:start -->"',
                'marker_end = "<!-- capability-snapshot:end -->"',
                "",
                "[labels]",
                'public_api_exports = "Portable API exports"',
                "",
            ]
        ),
    )
