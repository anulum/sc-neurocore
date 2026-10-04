# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real Rust documentation producer and refusal contracts

"""Verify the public Rust documentation CLI with actual Cargo and private ceilings."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def local_git_fixture_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep private native Git fixtures separate from the hosting job's checkout identity.

    Individual CI-event tests supply their own explicit event context after
    this fixture. Real Git sources and native producer executions are retained.
    """
    for name in ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_EVENT_PATH", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(name, raising=False)


REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCUMENTED = "//! A native documentation measurement fixture.\n/// Return one.\npub fn answer() -> u8 { 1 }\n"


def _project(root: Path, source: str, *, git: bool = True) -> Path:
    """Create a dependency-free Rust library with an actual compilable manifest."""
    root.mkdir()
    (root / "src").mkdir()
    (root / "Cargo.toml").write_text(
        '[package]\nname = "documentation_fixture"\nversion = "0.0.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    header = "\n".join(
        line.replace("#", "//", 1)
        for line in Path(__file__).read_text(encoding="utf-8").splitlines()[:7]
    )
    (root / "src/lib.rs").write_text(header + "\n" + source, encoding="utf-8")
    if git:
        (root / ".gitignore").write_text("native-target/\n")
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
    return root


def _run(
    root: Path, ceiling: Path, *, update: bool = False, search_path: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real public CLI with isolated native outputs and no fake executables."""
    assert shutil.which("cargo") is not None, "Actual Cargo is required for native validation"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = os.environ.get("CARGO_TARGET_DIR", str(root.parent / "native-target"))
    if search_path is not None:
        env["PATH"] = search_path
    argv = [
        sys.executable,
        "-B",
        "-m",
        "tools.rust_doc_ratchet",
        "--repo",
        str(root),
        "--manifest",
        "Cargo.toml",
        "--ceiling",
        str(ceiling),
    ]
    if update:
        argv.append("--update")
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
    )


def _ceiling(path: Path, debt: int) -> Path:
    """Retain a private input ceiling without changing the canonical debt record."""
    path.write_text(json.dumps({"undocumented": debt, "preserve": "original"}), encoding="utf-8")
    return path


def test_documented_library_passes_a_zero_ceiling(tmp_path: Path) -> None:
    """A successful actual compilation with full documentation reports zero debt."""
    root = _project(tmp_path / "documented", _DOCUMENTED)
    result = _run(root, _ceiling(tmp_path / "ceiling.json", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unchanged at 0" in result.stdout


def test_actual_missing_documentation_exceeds_a_zero_ceiling(tmp_path: Path) -> None:
    """Native missing_docs warnings must still produce a failing debt comparison."""
    root = _project(tmp_path / "undocumented", _DOCUMENTED.replace("/// Return one.\n", ""))
    result = _run(root, _ceiling(tmp_path / "ceiling.json", 0))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 undocumented" in result.stdout
    assert "rose" in result.stdout


@pytest.mark.parametrize("update", [False, True])
def test_compile_failure_never_passes_or_changes_the_ceiling(tmp_path: Path, update: bool) -> None:
    """Actual invalid Rust cannot become a zero measurement or lower a private baseline."""
    root = _project(
        tmp_path / "invalid", "//! Invalid Rust fixture.\npub fn answer() -> u8 { absent_name }\n"
    )
    ceiling = _ceiling(tmp_path / "ceiling.json", 10)
    original = ceiling.read_bytes()
    result = _run(root, ceiling, update=update)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cargo rustc" in result.stdout
    assert "error:" in result.stdout
    assert "debt fell" not in result.stdout
    assert ceiling.read_bytes() == original


@pytest.mark.parametrize("update", [False, True])
def test_missing_manifest_never_passes_or_changes_the_ceiling(tmp_path: Path, update: bool) -> None:
    """An absent Cargo manifest is an unavailable measurement rather than zero debt."""
    root = tmp_path / "missing"
    root.mkdir()
    ceiling = _ceiling(tmp_path / "ceiling.json", 10)
    original = ceiling.read_bytes()
    result = _run(root, ceiling, update=update)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "error:" in result.stdout
    assert ceiling.read_bytes() == original


def test_missing_cargo_is_an_actual_refusal(tmp_path: Path) -> None:
    """An empty executable search path must refuse without a native measurement."""
    root = _project(tmp_path / "missing_cargo", _DOCUMENTED)
    result = _run(root, _ceiling(tmp_path / "ceiling.json", 0), search_path="")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cargo is not installed" in result.stdout


def test_failed_source_revision_preserves_the_ceiling(tmp_path: Path) -> None:
    """An actual Git provenance failure must refuse an otherwise valid ceiling update."""
    root = _project(tmp_path / "no_revision", _DOCUMENTED, git=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert revision.returncode != 0, "This native refusal requires no usable source revision"
    ceiling = _ceiling(tmp_path / "ceiling.json", 10)
    original = ceiling.read_bytes()
    result = _run(root, ceiling, update=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "native Git Rust cohort" in result.stdout
    assert ceiling.read_bytes() == original
