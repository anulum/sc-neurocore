# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — actual documentation measurement producer outcomes

"""Exercise public documentation measurements against actual native tools and source errors."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD = """
import json
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
from tools.documentation_debt import measure_python, measure_rust, measure_typescript, measure_go
root = pathlib.Path(sys.argv[2])
language = sys.argv[3]
if language == "python":
    value = measure_python(root, json.loads(sys.argv[4]))
elif language == "rust":
    value = measure_rust(root, "Cargo.toml")
elif language == "typescript":
    value = measure_typescript(root, "eslint.measure.js")
else:
    value = measure_go(root)
print(json.dumps(value.to_public_dict()))
"""


def _measure(
    root: Path,
    language: str,
    *,
    scopes: list[str] | None = None,
    search_path: str | None = None,
    interpreter: str | None = None,
) -> dict[str, object]:
    """Call the public measurement API through an isolated interpreter and real tools."""
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = os.environ.get("CARGO_TARGET_DIR", str(root / "native-target"))
    env["NPM_CONFIG_CACHE"] = str(root / "native-npm-cache")
    env["NPM_CONFIG_OFFLINE"] = "true"
    if search_path is not None:
        env["PATH"] = search_path
    result = subprocess.run(
        [
            interpreter or sys.executable,
            "-I",
            "-B",
            "-c",
            _CHILD,
            str(REPO_ROOT),
            str(root),
            language,
            json.dumps(["sample.py"] if scopes is None else scopes),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report: dict[str, object] = json.loads(result.stdout)
    return report


def _unmeasured(report: dict[str, object]) -> None:
    """Require no numeric result and a concrete failure reason from the actual producer."""
    assert report["undocumented"] is None
    assert report["files"] is None
    assert report["not_measured_reason"]


@pytest.mark.parametrize("documented", [False, True])
def test_python_counts_actual_documentation_findings(tmp_path: Path, documented: bool) -> None:
    """Successful Ruff measurements distinguish real missing docs from a documented module."""
    doc = '    """Return one from the sample function."""\n' if documented else ""
    (tmp_path / "sample.py").write_text(
        '"""Actual Python measurement fixture."""\n\ndef answer() -> int:\n'
        + doc
        + "    return 1\n",
        encoding="utf-8",
    )
    report = _measure(tmp_path, "python")
    assert report["undocumented"] == (0 if documented else 1)
    assert report["files"] == (0 if documented else 1)
    assert report["tool_version"]


@pytest.mark.parametrize("input_case", ["missing", "syntax", "empty_scope"])
def test_python_invalid_inputs_are_unmeasured(tmp_path: Path, input_case: str) -> None:
    """Missing input, invalid Python and an empty declared cohort cannot masquerade as zero."""
    if input_case == "syntax":
        (tmp_path / "sample.py").write_text("def answer(:\n", encoding="utf-8")
    _unmeasured(_measure(tmp_path, "python", scopes=[] if input_case == "empty_scope" else None))


def test_missing_ruff_is_unmeasured_in_an_actual_interpreter(tmp_path: Path) -> None:
    """A real isolated system interpreter without Ruff cannot acquire a numeric result."""
    interpreter = "/usr/bin/python3"
    probe = subprocess.run(
        [
            interpreter,
            "-I",
            "-B",
            "-c",
            "import importlib.util; print(importlib.util.find_spec('ruff'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert probe.stdout.strip() == "None", (
        "This native refusal lane requires an actual Ruff-free interpreter"
    )
    (tmp_path / "sample.py").write_text('"""Documented sample."""\n', encoding="utf-8")
    _unmeasured(_measure(tmp_path, "python", interpreter=interpreter))


def _rust_project(root: Path, source: str) -> None:
    """Write a tiny actual Rust crate without dependencies or a simulated compiler."""
    assert shutil.which("cargo") is not None, "Actual Cargo is required for native validation"
    (root / "src").mkdir()
    (root / "Cargo.toml").write_text(
        '[package]\nname = "measurement_fixture"\nversion = "0.0.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (root / "src/lib.rs").write_text(source, encoding="utf-8")
    (root / ".gitignore").write_text("native-target/\n", encoding="utf-8")
    for argv in [
        ["git", "init", "-q", "--initial-branch=main"],
        ["cargo", "generate-lockfile", "--offline"],
        ["git", "add", "--", "."],
        [
            "git",
            "-c",
            "user.name=Native contract",
            "-c",
            "user.email=native@example.invalid",
            "commit",
            "-qm",
            "original native source",
        ],
    ]:
        subprocess.run(argv, cwd=root, capture_output=True, check=True, timeout=30)


@pytest.mark.parametrize("input_case", ["missing", "syntax", "missing_cargo"])
def test_rust_invalid_producers_are_unmeasured(tmp_path: Path, input_case: str) -> None:
    """Missing manifests, real compiler errors and absent Cargo retain unknown debt."""
    assert shutil.which("cargo") is not None, "Actual Cargo is required for native validation"
    if input_case != "missing":
        _rust_project(tmp_path, "//! Rust fixture.\npub fn answer() -> u8 { absent_name }\n")
    _unmeasured(
        _measure(tmp_path, "rust", search_path="" if input_case == "missing_cargo" else None)
    )


def test_rust_unrelated_warning_does_not_inflate_documentation_files(tmp_path: Path) -> None:
    """Real warnings in a second file must not enlarge the missing-docs file count."""
    _rust_project(
        tmp_path,
        "//! Rust documentation fixture.\n/// Hold a private helper.\npub mod hidden;\npub fn answer() -> u8 { 1 }\n",
    )
    (tmp_path / "src/hidden.rs").write_text("fn unused_helper() -> u8 { 2 }\n", encoding="utf-8")
    report = _measure(tmp_path, "rust")
    assert report["undocumented"] == 1
    assert report["files"] == 1
    assert report["tool_version"]


def _typescript_project(root: Path, source: str) -> None:
    """Use the actual installed ESLint environment and unchanged measurement config."""
    frontend = root / "studio/frontend"
    (frontend / "src").mkdir(parents=True)
    installed = REPO_ROOT / "studio/frontend/node_modules"
    assert (installed / ".bin/eslint").is_file(), "Actual installed ESLint is required"
    (frontend / "node_modules").symlink_to(installed, target_is_directory=True)
    (frontend / "package.json").write_text('{"type":"module","private":true}', encoding="utf-8")
    shutil.copyfile(REPO_ROOT / "studio/frontend/eslint.measure.js", frontend / "eslint.measure.js")
    (frontend / "src/sample.ts").write_text(source, encoding="utf-8")


@pytest.mark.parametrize("documented", [False, True])
def test_typescript_counts_actual_documentation_findings(tmp_path: Path, documented: bool) -> None:
    """Real ESLint distinguishes valid zero debt from an actual undocumented export."""
    _typescript_project(
        tmp_path,
        ("/** Return one from the sample function. */\n" if documented else "")
        + "export function answer(): number { return 1; }\n",
    )
    report = _measure(tmp_path, "typescript")
    assert report["undocumented"] == (0 if documented else 1)
    assert report["files"] == (0 if documented else 1)
    assert report["tool_version"]


def test_typescript_parse_failure_is_unmeasured(tmp_path: Path) -> None:
    """An actual TypeScript parser error cannot become a successful zero-docs figure."""
    _typescript_project(tmp_path, "export function answer( {\n")
    _unmeasured(_measure(tmp_path, "typescript"))


def test_missing_eslint_launcher_is_unmeasured(tmp_path: Path) -> None:
    """Missing real npx on an empty PATH records unavailable ESLint without a number."""
    _typescript_project(tmp_path, "export function answer(): number { return 1; }\n")
    _unmeasured(_measure(tmp_path, "typescript", search_path=""))


@pytest.mark.parametrize("valid", [False, True])
def test_go_parser_keeps_real_failure_distinct_from_zero(tmp_path: Path, valid: bool) -> None:
    """The real Go parser retains its existing rejection and documented-zero contract."""
    assert shutil.which("go") is not None, "Actual Go is required for native validation"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=10)
    source = (
        "// Package sample is documented.\npackage sample\n// Answer returns one.\nfunc Answer() int { return 1 }\n"
        if valid
        else "package !!!\n"
    )
    (tmp_path / "sample.go").write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "sample.go"], cwd=tmp_path, check=True, timeout=10)
    shutil.copytree(REPO_ROOT / "tools/godoc_coverage", tmp_path / "tools/godoc_coverage")
    report = _measure(tmp_path, "go")
    if valid:
        assert report["undocumented"] == 0
        assert report["files"] == 0
    else:
        _unmeasured(report)


def test_go_failed_source_revision_preserves_the_ceiling(tmp_path: Path) -> None:
    """The actual Go CLI must preserve a private ceiling when Git has no source revision."""
    assert shutil.which("go") is not None, "Actual Go is required for native validation"
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=10)
    (tmp_path / "sample.go").write_text(
        "// Package sample is documented.\npackage sample\n// Answer returns one.\nfunc Answer() int { return 1 }\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "sample.go"], cwd=tmp_path, check=True, timeout=10)
    shutil.copytree(REPO_ROOT / "tools/godoc_coverage", tmp_path / "tools/godoc_coverage")
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text('{"undocumented":10,"preserve":"original"}', encoding="utf-8")
    original = ceiling.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.go_doc_ratchet",
            "--repo",
            str(tmp_path),
            "--ceiling",
            str(ceiling),
            "--update",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source revision" in result.stdout
    assert ceiling.read_bytes() == original
