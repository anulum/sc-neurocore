# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API documentation generator tests

from pathlib import Path

import pytest

from scripts import generate_docs
from scripts.generate_docs import (
    build_markdown,
    check_markdown,
    generate_markdown,
    main,
)


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


def test_generate_markdown_excludes_private_symbols(tmp_path: Path) -> None:
    """The public API reference must omit underscore-prefixed symbols."""
    src_dir = tmp_path / "src" / "sc_neurocore"
    package_dir = src_dir / "core"
    package_dir.mkdir(parents=True)
    (package_dir / "surface.py").write_text(
        '"""Surface module."""\n\n'
        "class PublicApi:\n"
        '    """Documented public class."""\n'
        "    def run(self) -> None:\n"
        '        """Run the API."""\n'
        "    def _prepare(self) -> None:\n"
        '        """Prepare an internal state."""\n\n'
        "class _InternalApi:\n"
        '    """Internal class."""\n\n'
        "def public_function() -> None:\n"
        '    """Documented public function."""\n\n'
        "def _private_function() -> None:\n"
        '    """Internal function."""\n',
        encoding="utf-8",
    )

    rendered = build_markdown(src_dir)

    assert "PublicApi" in rendered
    assert "**run**()" in rendered
    assert "public_function" in rendered
    assert "_prepare" not in rendered
    assert "_InternalApi" not in rendered
    assert "_private_function" not in rendered


def _write_documented_source(tmp_path: Path) -> Path:
    """Create a minimal documented ``sc_neurocore`` source tree for tests."""
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
    return src_dir


def test_build_markdown_is_deterministic(tmp_path: Path) -> None:
    """Repeated builds of the same tree must be byte-identical (drift-gate safe)."""
    src_dir = _write_documented_source(tmp_path)

    first = build_markdown(src_dir)
    second = build_markdown(src_dir)

    assert first == second
    assert first.endswith("\n")
    assert "Module `core.stable`" in first


def test_check_markdown_detects_fresh_stale_and_missing(tmp_path: Path) -> None:
    """The freshness check must accept an up-to-date file and reject drift/absence."""
    src_dir = _write_documented_source(tmp_path)
    output = tmp_path / "API_REFERENCE.md"

    assert check_markdown(src_dir, output) is False  # not yet generated

    generate_markdown(src_dir, output)
    assert check_markdown(src_dir, output) is True

    output.write_text(output.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
    assert check_markdown(src_dir, output) is False


def test_repo_api_reference_stays_fresh() -> None:
    """The committed docs/API_REFERENCE.md must match the generator (the CI gate)."""
    assert main(["--check"]) == 0


def test_main_reports_stale_reference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` must return a non-zero code and an error line when stale."""
    monkeypatch.setattr(generate_docs, "check_markdown", lambda src, out: False)

    exit_code = main(["--check"])

    assert exit_code == 1
    assert "::error::" in capsys.readouterr().out


def test_main_regenerates_reference_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default action must regenerate the reference at the resolved paths."""
    src_dir = _write_documented_source(tmp_path)
    output = tmp_path / "API_REFERENCE.md"
    monkeypatch.setattr(generate_docs, "_default_paths", lambda: (src_dir, output))

    exit_code = main([])

    assert exit_code == 0
    assert output.exists()
    assert "SC-NeuroCore API Reference" in output.read_text(encoding="utf-8")
