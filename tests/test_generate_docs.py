# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API documentation generator tests

"""Verify public source extraction, call signatures and API-reference freshness."""

from pathlib import Path

import pytest

from scripts import generate_docs
from scripts.generate_docs import (
    build_markdown,
    check_markdown,
    generate_markdown,
    main,
    parse_file,
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


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("def dispatch(*, name, value):", ["*", "name", "value"]),
        ("def dispatch(value, /):", ["value", "/"]),
        (
            "def dispatch(value, /, fallback, *, name):",
            ["value", "/", "fallback", "*", "name"],
        ),
        (
            "def dispatch(value, *items, name=None, **labels):",
            ["value", "*items", "name", "**labels"],
        ),
        ("def dispatch(**options):", ["**options"]),
        ("def dispatch(self, *, value):", ["self", "*", "value"]),
        ("def dispatch():", []),
    ],
    ids=[
        "required-keyword-only",
        "positional-only",
        "mixed-positional-and-keyword",
        "variadic-with-keyword",
        "variadic-keyword",
        "module-parameter-named-self",
        "no-parameters",
    ],
)
def test_generated_function_signatures_preserve_parameter_kinds(
    tmp_path: Path, declaration: str, expected: list[str]
) -> None:
    """All Python parameter kinds survive public parsing and file generation."""
    source = tmp_path / "dispatch.py"
    source.write_text(declaration + '\n    """Dispatch the supplied values."""\n', encoding="utf-8")

    classes, functions = parse_file(source)
    output = tmp_path / "reference.md"
    generate_markdown(tmp_path, output)

    assert classes == []
    assert len(functions) == 1
    assert functions[0]["args"] == expected
    assert f"### Function `dispatch({', '.join(expected)})`" in output.read_text(encoding="utf-8")


def test_generated_method_signatures_preserve_bound_and_static_contracts(
    tmp_path: Path,
) -> None:
    """Method headers omit bound receivers and retain static parameters and markers."""
    source = tmp_path / "collector.py"
    source.write_text(
        '''class Collector:
    """Collect explicitly labelled samples."""

    def __init__(self, /, *, name):
        """Create a named collection."""

    def submit(self, sample, /, channel, *items, required, **options):
        """Submit one sample and optional additional items."""

    @classmethod
    def create(cls, /, *, name):
        """Construct a collection by name."""

    @staticmethod
    def decode(self, /, *, mode):
        """Decode a payload whose parameter happens to be named self."""

    def reset(self):
        """Discard collected samples."""

    def keyword(self, *, value):
        """Record a keyword-only value."""

    def variadic(*items):
        """Receive the bound instance and any supplied items."""
''',
        encoding="utf-8",
    )

    classes, functions = parse_file(source)
    rendered = build_markdown(tmp_path)

    assert functions == []
    assert len(classes) == 1
    assert {method["name"]: method["args"] for method in classes[0]["methods"]} == {
        "__init__": ["*", "name"],
        "submit": ["sample", "/", "channel", "*items", "required", "**options"],
        "create": ["*", "name"],
        "decode": ["self", "/", "*", "mode"],
        "reset": [],
        "keyword": ["*", "value"],
        "variadic": ["*items"],
    }
    assert r"- **__init__**(\*, name)" in rendered
    assert r"- **submit**(sample, /, channel, \*items, required, \*\*options)" in rendered
    assert r"- **create**(\*, name)" in rendered
    assert r"- **decode**(self, /, \*, mode)" in rendered
    assert "- **reset**()" in rendered


@pytest.mark.parametrize(
    "name",
    ["require_integral", "require_bool", "require_finite_float", "require_u32_seed"],
)
def test_real_validation_headers_include_required_keyword_arguments(name: str) -> None:
    """The actual validation declarations and committed headers retain name and value."""
    repo = Path(__file__).resolve().parents[1]
    source = repo / "src/sc_neurocore/_native/learning_validation.py"

    _, functions = parse_file(source)

    function = next(item for item in functions if item["name"] == name)
    assert function["args"] == ["*", "name", "value"]
    assert f"### Function `{name}(*, name, value)`" in (repo / "docs/API_REFERENCE.md").read_text(
        encoding="utf-8"
    )
