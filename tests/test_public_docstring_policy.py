# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — scoped public docstring policy tests

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCSTRING_POLICY = REPO_ROOT / "docs" / "docstring_policy.toml"


def _public_definitions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    definitions: list[tuple[str, ast.AST]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            definitions.append((node.name, node))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_"):
                        definitions.append((f"{node.name}.{item.name}", item))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                definitions.append((node.name, node))

    return definitions


def test_scoped_public_python_files_have_maintained_docstrings() -> None:
    """Policy-listed public Python files must document their public API surface."""
    policy = tomllib.loads(DOCSTRING_POLICY.read_text(encoding="utf-8"))
    min_chars = policy["quality"]["min_docstring_chars"]
    expected_files = policy["quality"]["expected_file_count"]

    violations: list[str] = []
    policy_files = policy["file"]
    assert len(policy_files) == expected_files

    for entry in policy_files:
        path = REPO_ROOT / entry["path"]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        module_doc = ast.get_docstring(tree)
        if entry.get("require_module_docstring", True) and (
            module_doc is None or len(module_doc.strip()) < min_chars
        ):
            violations.append(f"{entry['path']}: module docstring")

        allowed_missing = set(entry.get("allow_missing", []))
        for name, node in _public_definitions(tree):
            if name in allowed_missing:
                continue
            doc = ast.get_docstring(node)  # type: ignore[arg-type]
            if doc is None or len(doc.strip()) < min_chars:
                violations.append(f"{entry['path']}: {name}")

    assert violations == []


def test_scoped_public_files_pass_numpy_docstring_rules() -> None:
    """Policy-listed files must satisfy ruff `D` rules under the NumPy convention.

    This promotes the scoped policy from a minimum-length floor to enforcing the
    NumPy-convention docstring rules mandated by the 2026-06-17 strict-typing and
    docstring broadcast. The enforced surface is exactly ``docs/docstring_policy.toml``;
    the file list grows package-by-package until ``D`` can be promoted to the global
    ruff ``select``.
    """
    policy = tomllib.loads(DOCSTRING_POLICY.read_text(encoding="utf-8"))
    files = [entry["path"] for entry in policy["file"]]

    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "D", "--no-cache", *files],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "ruff NumPy-convention docstring violations in policy-scoped files:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
