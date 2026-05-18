# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the maintained Markdown API reference from Python docstrings."""

import ast
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


def parse_file(filepath: str | Path) -> tuple[list[ClassDoc], list[FunctionDoc]]:
    """Parse public classes and functions from one Python source file."""
    filepath = Path(filepath)
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))

    classes: list[ClassDoc] = []
    functions: list[FunctionDoc] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods: list[MethodDoc] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
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


def generate_markdown(src_dir: str | Path, output_file: str | Path) -> None:
    """Generate the Markdown API reference for a source directory."""
    src_path = Path(src_dir)
    output_path = Path(output_file)
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

    output_path.write_text(f"{md.rstrip()}\n", encoding="utf-8")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src" / "sc_neurocore"
    out = repo_root / "docs" / "API_REFERENCE.md"
    generate_markdown(src, out)
