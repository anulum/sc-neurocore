# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


"""Generate the maintained Markdown API reference from Python docstrings."""

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict


EXCLUDED_SOURCE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".nox",
        ".pixi",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "site",
        "venv",
    }
)


class MethodDoc(TypedDict):
    """Structured documentation extracted from a public class method."""

    name: str
    args: list[str]
    doc: str | None


class ClassDoc(TypedDict):
    """Structured documentation extracted from a public class definition."""

    name: str
    doc: str | None
    methods: list[MethodDoc]


class FunctionDoc(TypedDict):
    """Structured documentation extracted from a public module function."""

    name: str
    args: list[str]
    doc: str | None


def _is_public_name(name: str) -> bool:
    """Return whether ``name`` belongs in the public API reference."""
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def parse_file(filepath: str | Path) -> tuple[list[ClassDoc], list[FunctionDoc]]:
    """Parse public classes and functions from one Python source file."""
    filepath = Path(filepath)
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))

    classes: list[ClassDoc] = []
    functions: list[FunctionDoc] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not _is_public_name(node.name):
                continue
            methods: list[MethodDoc] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not _is_public_name(item.name):
                        continue
                    doc = ast.get_docstring(item)
                    methods.append(
                        {
                            "name": item.name,
                            "args": [a.arg for a in item.args.args if a.arg != "self"],
                            "doc": doc,
                        }
                    )
            doc = ast.get_docstring(node)
            classes.append({"name": node.name, "doc": doc, "methods": methods})
        elif isinstance(node, ast.FunctionDef):
            if not _is_public_name(node.name):
                continue
            doc = ast.get_docstring(node)
            functions.append(
                {"name": node.name, "args": [a.arg for a in node.args.args], "doc": doc}
            )

    return classes, functions


def _iter_source_files(src_dir: str | Path) -> list[Path]:
    src_path = Path(src_dir)
    files: list[Path] = []

    for filepath in src_path.rglob("*.py"):
        relative_parts = filepath.relative_to(src_path).parts
        if filepath.name == "__init__.py":
            continue
        if any(part in EXCLUDED_SOURCE_DIRS for part in relative_parts[:-1]):
            continue
        files.append(filepath)

    return sorted(files)


def _markdown_text(text: str) -> str:
    return text.replace("[", "&#91;").replace("]", "&#93;")


def build_markdown(src_dir: str | Path) -> str:
    """Build and return the Markdown API reference for a source directory.

    Parameters
    ----------
    src_dir : str or pathlib.Path
        Root of the ``sc_neurocore`` package source tree.

    Returns
    -------
    str
        The full Markdown document, terminated by a single trailing newline.
    """
    src_path = Path(src_dir)
    md = "# SC-NeuroCore API Reference\n\n"

    for filepath in _iter_source_files(src_path):
        rel_path = filepath.relative_to(src_path)
        module_name = ".".join(rel_path.with_suffix("").parts)

        classes, functions = parse_file(filepath)

        if not classes and not functions:
            continue

        md += f"## Module `{module_name}`\n\n"

        for cls in classes:
            md += f"### Class `{cls['name']}`\n"
            if cls["doc"]:
                md += f"{_markdown_text(cls['doc'])}\n\n"

            for method in cls["methods"]:
                args = ", ".join(method["args"])
                md += f"- **{method['name']}**({args})\n"
                if method["doc"]:
                    md += f"  - {_markdown_text(method['doc'].splitlines()[0])}\n"
            md += "\n"

        for func in functions:
            args = ", ".join(func["args"])
            md += f"### Function `{func['name']}({args})`\n"
            if func["doc"]:
                md += f"{_markdown_text(func['doc'])}\n\n"

        md += "---\n\n"

    return f"{md.rstrip()}\n"


def generate_markdown(src_dir: str | Path, output_file: str | Path) -> None:
    """Generate the Markdown API reference for a source directory."""
    output_path = Path(output_file)
    output_path.write_text(build_markdown(src_dir), encoding="utf-8")
    print(f"Generated {output_path}")


def check_markdown(src_dir: str | Path, output_file: str | Path) -> bool:
    """Return whether ``output_file`` matches freshly generated documentation.

    Parameters
    ----------
    src_dir : str or pathlib.Path
        Root of the ``sc_neurocore`` package source tree.
    output_file : str or pathlib.Path
        The committed Markdown API reference to compare against.

    Returns
    -------
    bool
        ``True`` when the committed file exists and is byte-identical to the
        freshly generated document, ``False`` otherwise.
    """
    output_path = Path(output_file)
    if not output_path.exists():
        return False
    return output_path.read_text(encoding="utf-8") == build_markdown(src_dir)


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "src" / "sc_neurocore", repo_root / "docs" / "API_REFERENCE.md"


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the Markdown API reference or check that it is up to date.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv``). ``--check`` verifies
        freshness without writing; the default action regenerates the file.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` when ``--check`` finds the
        committed reference stale.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/API_REFERENCE.md is stale instead of regenerating it.",
    )
    args = parser.parse_args(argv)

    src, out = _default_paths()
    if args.check:
        if check_markdown(src, out):
            print(f"{out} is up to date")
            return 0
        print(
            f"::error::{out} is stale; run `python scripts/generate_docs.py` and commit the result"
        )
        return 1

    generate_markdown(src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
