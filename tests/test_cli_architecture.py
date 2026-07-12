# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CLI package architecture tests

"""Protect the command package boundary and removal of false accelerator mirrors."""

from __future__ import annotations

import ast
from pathlib import Path

from sc_neurocore.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = REPO_ROOT / "src/sc_neurocore/cli"
COMMAND_ROOT = CLI_ROOT / "commands"
MAX_COMMAND_LINES = 700
MAX_COMPOSITION_ROOT_LINES = 150


def test_cli_entrypoint_is_packaged_without_legacy_module() -> None:
    assert callable(main)
    assert CLI_ROOT.is_dir()
    assert not (REPO_ROOT / "src/sc_neurocore/cli.py").exists()
    assert (CLI_ROOT / "__main__.py").is_file()


def test_command_modules_have_one_registration_boundary() -> None:
    command_files = sorted(COMMAND_ROOT.glob("*.py"))
    assert {path.stem for path in command_files} == {
        "__init__",
        "compile",
        "deploy",
        "formal",
        "hub",
        "info",
        "maintenance",
        "mapping",
        "scnir",
        "serve",
        "studio",
        "synthesis",
    }
    for path in command_files:
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        registrations = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("add_")
        ]
        assert len(registrations) == 1, path


def test_cli_split_cannot_regrow_a_godfile() -> None:
    """Keep the parser small and each command below the audited monolith threshold."""
    parser_lines = (CLI_ROOT / "parser.py").read_text(encoding="utf-8").count("\n") + 1
    assert parser_lines <= MAX_COMPOSITION_ROOT_LINES
    for path in COMMAND_ROOT.glob("*.py"):
        command_lines = path.read_text(encoding="utf-8").count("\n") + 1
        assert command_lines <= MAX_COMMAND_LINES, path


def test_command_dependency_graph_is_acyclic() -> None:
    dependencies: dict[str, set[str]] = {}
    for path in COMMAND_ROOT.glob("*.py"):
        module = path.stem
        dependencies[module] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependencies[module].add(node.module.split(".", maxsplit=1)[0])

    visited: set[str] = set()
    active: set[str] = set()

    def visit(module: str) -> None:
        if module in active:
            raise AssertionError(f"CLI command import cycle at {module}")
        if module in visited:
            return
        active.add(module)
        for dependency in dependencies.get(module, set()):
            visit(dependency)
        active.remove(module)
        visited.add(module)

    for module in dependencies:
        visit(module)


def test_non_compute_cli_has_no_false_polyglot_mirrors() -> None:
    stale_mirrors = (
        "src/sc_neurocore/accel/julia/core/cli.jl",
        "src/sc_neurocore/accel/mojo/kernels/cli.mojo",
        "src/sc_neurocore/accel/rust/safety/cli.rs",
    )
    assert all(not (REPO_ROOT / relative_path).exists() for relative_path in stale_mirrors)
