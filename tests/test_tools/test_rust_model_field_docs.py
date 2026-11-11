# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust model field documentation from the model descriptors

"""Exercise the descriptor-to-rustdoc field documentation tool on real files."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tools.rust_model_field_docs import (
    DESCRIPTOR_DIRECTORY,
    ENGINE_SOURCE,
    REPO_ROOT,
    FieldDocError,
    doc_text,
    main,
    plan,
)

_DESCRIPTOR = """
[metadata]
class_name = "Model"

[parameters.a]
meaning = "already documented in Rust"
unit = "mV"

[parameters.b]
meaning = "leak reversal potential"
unit = "mV"

[parameters.shared]
meaning = "the parameter meaning"
unit = "ms"

[state.c]
meaning = "recovery variable"
unit = "1"

[state.shared]
meaning = "the state meaning"

[state.d]
init = 0.0

[parameters.private]
meaning = "never public"
"""

_SOURCE = """// Fixture engine source.
#[derive(Clone)]
pub struct Model {
    /// Already documented.
    pub a: f64,
    #[allow(dead_code)]
    pub b: f64,
    pub c: f64,
    pub shared: f64,
    pub d: f64, // trailing author comment
    private: f64,
}

pub struct Other {
    pub b: f64,
}
"""

_DOCUMENTED = """// Fixture engine source.
#[derive(Clone)]
pub struct Model {
    /// Already documented.
    pub a: f64,
    /// Leak reversal potential (mV).
    #[allow(dead_code)]
    pub b: f64,
    /// Recovery variable (dimensionless).
    pub c: f64,
    /// The parameter meaning (ms).
    pub shared: f64,
    pub d: f64, // trailing author comment
    private: f64,
}

pub struct Other {
    pub b: f64,
}
"""


def _repository(tmp_path: Path, source: str = _SOURCE, descriptor: str = _DESCRIPTOR) -> Path:
    """Create a repository layout with one descriptor and one engine source."""
    (tmp_path / DESCRIPTOR_DIRECTORY).mkdir(parents=True)
    (tmp_path / DESCRIPTOR_DIRECTORY / "Model.toml").write_text(descriptor, encoding="utf-8")
    (tmp_path / ENGINE_SOURCE / "neurons").mkdir(parents=True)
    (tmp_path / ENGINE_SOURCE / "neurons" / "model.rs").write_text(source, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("meaning", "unit", "expected"),
    [
        ("leak conductance", "mS/cm^2", "Leak conductance (mS/cm^2)."),
        ("sigmoid gain", "1", "Sigmoid gain (dimensionless)."),
        ("threshold.", None, "Threshold."),
        ("threshold", "  ", "Threshold."),
        ("[Na]_i ratio * k", "1", "\\[Na\\]_i ratio \\* k (dimensionless)."),
        ("Λ-scaled rate", "1/ms", "Λ-scaled rate (1/ms)."),
    ],
)
def test_doc_text_renders_meaning_and_unit(meaning: str, unit: str | None, expected: str) -> None:
    """Meanings are capitalised and terminated; unit 1 is dimensionless; Markdown is escaped."""
    assert doc_text(meaning, unit) == expected


def test_an_empty_meaning_is_refused() -> None:
    """A descriptor entry without words cannot become documentation."""
    with pytest.raises(FieldDocError):
        doc_text(" . ", "mV")


def test_check_lists_and_write_documents_only_undocumented_described_public_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Documented, private, undescribed and other-struct fields are left exactly as they are."""
    root = _repository(tmp_path)
    planned = plan(root)
    assert [(item.field, item.line, item.insert_at) for item in planned] == [
        ("b", 7, 6),
        ("c", 8, 8),
        ("shared", 9, 9),
    ]
    assert main(["--repo", str(root), "--check"]) == 1
    output = capsys.readouterr().out
    assert "engine/src/neurons/model.rs:7: Model.b lacks its descriptor doc" in output
    assert "3 described Rust fields lack documentation." in output
    assert main(["--repo", str(root)]) == 0
    assert "Documented 3 Rust fields" in capsys.readouterr().out
    assert (root / ENGINE_SOURCE / "neurons" / "model.rs").read_text(
        encoding="utf-8"
    ) == _DOCUMENTED
    assert main(["--repo", str(root), "--check"]) == 0
    assert main(["--repo", str(root)]) == 0
    assert (root / ENGINE_SOURCE / "neurons" / "model.rs").read_text(
        encoding="utf-8"
    ) == _DOCUMENTED


def test_a_doc_attribute_counts_as_documentation_and_no_final_newline_is_kept(
    tmp_path: Path,
) -> None:
    """A ``#[doc = ...]`` attribute documents the field; a source without a final newline keeps none."""
    source = 'pub struct Model {\n    #[doc = "Documented."]\n    pub b: f64,\n    pub c: f64,\n}'
    root = _repository(tmp_path, source)
    assert [item.field for item in plan(root)] == ["c"]
    assert main(["--repo", str(root)]) == 0
    written = (root / ENGINE_SOURCE / "neurons" / "model.rs").read_text(encoding="utf-8")
    assert written.endswith("    pub c: f64,\n}")
    assert "    /// Recovery variable (dimensionless).\n    pub c: f64," in written


@pytest.mark.parametrize(
    ("descriptor", "source", "message"),
    [
        ("[metadata]\nname = 'Model'\n", _SOURCE, "names no class"),
        ("[metadata\nclass_name = 'Model'\n", _SOURCE, "Cannot read descriptor"),
        (_DESCRIPTOR, _SOURCE + _SOURCE, "is defined in"),
        (_DESCRIPTOR, "pub struct Model {\n    pub b: f64,\n", "not closed"),
        (
            _DESCRIPTOR.replace('meaning = "recovery variable"', 'meaning = ""'),
            _SOURCE,
            "must not be empty",
        ),
    ],
)
def test_unreadable_or_ambiguous_inputs_are_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    descriptor: str,
    source: str,
    message: str,
) -> None:
    """Every input the tool cannot read unambiguously stops it with status 2 and writes nothing."""
    root = _repository(tmp_path, source, descriptor)
    assert main(["--repo", str(root)]) == 2
    assert message in capsys.readouterr().err
    assert (root / ENGINE_SOURCE / "neurons" / "model.rs").read_text(encoding="utf-8") == source


def test_two_descriptors_of_one_class_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One class may have one descriptor, or the field meaning would be ambiguous."""
    root = _repository(tmp_path)
    (root / DESCRIPTOR_DIRECTORY / "Copy.toml").write_text(_DESCRIPTOR, encoding="utf-8")
    assert main(["--repo", str(root), "--check"]) == 2
    assert "Two descriptors describe the class Model." in capsys.readouterr().err


def test_the_command_line_entry_point_reports_through_its_exit_status(tmp_path: Path) -> None:
    """The module runs as a program and fails its check while fields lack documentation."""
    root = _repository(tmp_path)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tools.rust_model_field_docs", "--repo", str(root), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 1
    assert "3 described Rust fields lack documentation." in result.stdout
