# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API documentation generator tests

from pathlib import Path

import pytest

from scripts.generate_docs import generate_markdown


def test_generate_markdown_skips_embedded_tool_environments(tmp_path: Path) -> None:
    """Embedded tool environments under source trees must not enter API docs."""
    src_dir = tmp_path / "src" / "sc_neurocore"
    package_dir = src_dir / "core"
    package_dir.mkdir(parents=True)
    (package_dir / "stable.py").write_text(
        '"""Stable public module."""\n\n'
        "class StableApi:\n"
        '    """Documented public class."""\n'
        "    def run(self) -> None:\n"
        '        """Run the API."""\n',
        encoding="utf-8",
    )

    embedded = src_dir / "accel" / "mojo" / "venv" / "lib" / "python3.14"
    embedded.mkdir(parents=True)
    (embedded / "python314_only.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )

    output = tmp_path / "API_REFERENCE.md"

    generate_markdown(src_dir, output)

    rendered = output.read_text(encoding="utf-8")
    assert "Module `core.stable`" in rendered
    assert "StableApi" in rendered
    assert ".pixi" not in rendered
    assert "python314_only" not in rendered


def test_generate_markdown_fails_on_invalid_product_source(tmp_path: Path) -> None:
    """Invalid first-party source must fail generation instead of being hidden."""
    src_dir = tmp_path / "src" / "sc_neurocore"
    package_dir = src_dir / "core"
    package_dir.mkdir(parents=True)
    (package_dir / "broken.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError):
        generate_markdown(src_dir, tmp_path / "API_REFERENCE.md")


def test_generate_markdown_escapes_docstring_reference_syntax(
    tmp_path: Path,
) -> None:
    """Generated docstrings must not create unresolved MkDocs references."""
    src_dir = tmp_path / "src" / "sc_neurocore"
    package_dir = src_dir / "core"
    package_dir.mkdir(parents=True)
    (package_dir / "matrix.py").write_text(
        '"""Matrix helpers."""\n\n'
        "def connect() -> None:\n"
        '    """2D weight matrix [src_neuron][dst_neuron]."""\n',
        encoding="utf-8",
    )

    output = tmp_path / "API_REFERENCE.md"

    generate_markdown(src_dir, output)

    rendered = output.read_text(encoding="utf-8")
    assert "&#91;src_neuron&#93;&#91;dst_neuron&#93;" in rendered
    assert "[dst_neuron]" not in rendered
