# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — original native Rust documentation debt contracts

"""Exercise original Git Rust debt and the actual public ratchet CLI."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import runpy
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


from tests.test_tools.test_rust_doc_measurement_native import rust_project
from tools.rust_doc_history import INDIVIDUAL_SCHEMA, LEGACY_SCHEMA, protect_rust_debt
from tools.rust_doc_measurement import RustMeasurementError, measure_rust_findings
from tools.rust_doc_symbols import NativeParser, build_parser
from tools.rust_doc_ratchet import main

REPO = Path(__file__).resolve().parents[2]
OLD = "//! Native original crate.\npub fn old() {}\n"


@pytest.fixture(scope="module")
def parser() -> NativeParser:
    """Reuse an actual compiled native syntax producer across original debt contracts."""
    return build_parser(REPO)


def _git(root: Path, *args: str) -> str:
    """Require successful actual fixture Git operations."""
    return subprocess.check_output(["git", *args], cwd=root, text=True, timeout=30).strip()


def _baseline(
    root: Path, parser: NativeParser, source: str = OLD, *, extra: dict[str, str] | None = None
) -> tuple[Path, Path]:
    """Commit actual compiler-measured original declaration allowances."""
    root = rust_project(root, source, extra=extra)
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    ceiling = root / "ceiling.json"
    ceiling.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_SCHEMA,
                "undocumented": measured.undocumented,
                "undocumented_files": measured.files,
                "provenance": {"rustc": measured.rustc_version},
            }
        )
    )
    _git(root, "add", "--", "ceiling.json")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original measured ceiling",
    )
    return root, ceiling


def _cli(
    root: Path, ceiling: Path, *, update: bool = False, manifest: str = "Cargo.toml"
) -> subprocess.CompletedProcess[str]:
    """Exercise the public ratchet with native Cargo and isolated artifact output."""
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = os.environ.get("CARGO_TARGET_DIR", str(root.parent / "native-target"))
    argv = [
        sys.executable,
        "-B",
        "-m",
        "tools.rust_doc_ratchet",
        "--repo",
        str(root),
        "--manifest",
        manifest,
        "--ceiling",
        str(ceiling),
    ]
    if update:
        argv.append("--update")
    result = subprocess.run(
        argv, cwd=REPO, env=env, capture_output=True, text=True, check=False, timeout=90
    )
    with redirect_stdout(StringIO()):
        direct_code = main(argv[4:])
    assert direct_code == result.returncode
    return result


def test_original_native_cases_are_protected(parser: NativeParser, tmp_path: Path) -> None:
    """An unchanged compiler case passes the public CLI with independently measured originals."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    protected = protect_rust_debt(root, measured, ceiling, parser=parser)
    assert protected.original.cases == measured.cases
    assert protected.original_ceiling == 1
    assert protected.revision == _git(root, "rev-parse", "HEAD")
    assert protected.to_public_dict()["original"] == protected.original.to_public_dict()
    result = _cli(root, ceiling)
    assert result.returncode == 0, result.stdout + result.stderr


MACRO = """//! Native original crate.
macro_rules! generate {
    ($name:ident) => {
        pub fn $name() {}
    };
}
generate!(first);
generate!(second);
"""


def test_original_macro_generated_debt_is_counted_and_repaid_only_by_documentation(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Original macro-generated debt reproduces its ceiling; current source must document it.

    The baseline predates the refusal of undocumented macro-generated declarations.
    Its two generated functions count under expansion identities, the unchanged
    current source is still refused, and documenting the macro body repays both.
    """
    root = rust_project(tmp_path / "repo", MACRO)
    ceiling = root / "ceiling.json"
    ceiling.write_text(json.dumps({"schema_version": LEGACY_SCHEMA, "undocumented": 2}))
    _git(root, "add", "--", "ceiling.json")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original macro-generated debt",
    )
    with pytest.raises(RustMeasurementError, match="could not be uniquely resolved"):
        measure_rust_findings(root, "Cargo.toml", parser=parser)
    refused = _cli(root, ceiling)
    assert refused.returncode == 2 and "could not be uniquely resolved" in refused.stdout
    (root / "src/lib.rs").write_text(
        MACRO.replace(
            "        pub fn $name", "        /// Generated function.\n        pub fn $name"
        )
    )
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 0
    protected = protect_rust_debt(root, measured, ceiling, parser=parser)
    identities = {case.identity for case in protected.original.cases}
    assert protected.original.undocumented == 2 and len(identities) == 2
    assert all(identity.startswith("/expansion:generate:") for identity in identities)
    repaid = _cli(root, ceiling)
    assert repaid.returncode == 0 and "fell: 0 undocumented" in repaid.stdout
    ceiling.write_text(json.dumps({"schema_version": LEGACY_SCHEMA, "undocumented": 1}))
    lowered = _cli(root, ceiling)
    assert lowered.returncode == 0, lowered.stdout + lowered.stderr


def test_original_image_ignores_generated_sources_of_the_default_target(tmp_path: Path) -> None:
    """Build outputs of the original image never enter its source cohort.

    Without a redirected target directory Cargo builds inside the checked-out
    image. A real build script generates Rust there, as dependencies of the
    engine do. The generated file must not count as a changed measurement input.
    The user's global Git excludes are switched off, as on a hosted runner, so
    that only the image's own ignore rules decide.
    """
    root = rust_project(tmp_path / "repo", OLD)
    (root / "build.rs").write_text(
        "fn main() {\n"
        '    let out = std::env::var("OUT_DIR").unwrap();\n'
        '    let path = std::path::Path::new(&out).join("generated.rs");\n'
        '    std::fs::write(path, "pub fn generated() {}\\n").unwrap();\n'
        "}\n"
    )
    (root / ".gitignore").write_text("native-target/\ntarget/\n")
    ceiling = root / "ceiling.json"
    ceiling.write_text(json.dumps({"schema_version": LEGACY_SCHEMA, "undocumented": 1}))
    _git(root, "add", "--", "build.rs", ".gitignore", "ceiling.json")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original crate with a generating build script",
    )
    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        [
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
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unchanged at 1" in result.stdout
    assert list((root / "target").rglob("generated.rs"))


@pytest.mark.parametrize("kind", ["function", "receiver", "field"])
def test_same_count_cannot_trade_individual_debt(
    parser: NativeParser, tmp_path: Path, kind: str
) -> None:
    """Equal native counts cannot replace original debt with another declaration or owner."""
    source = OLD
    candidate = (
        "//! Native original crate.\n/// Original function.\npub fn old() {}\npub fn new() {}\n"
    )
    if kind == "receiver":
        source = "//! Native crate.\n/// Original type.\npub struct Old;\nimpl Old { pub fn record(&self) {} }\n"
        candidate = source.replace("Old", "New")
    elif kind == "field":
        source = "//! Native crate.\n/// Original type.\npub struct Old { pub value: u8 }\n"
        candidate = source.replace("Old", "New")
    root, ceiling = _baseline(tmp_path / "repo", parser, source)
    original = ceiling.read_bytes()
    (root / "src/lib.rs").write_text(candidate)
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 1
    with pytest.raises(RustMeasurementError, match="New undocumented"):
        protect_rust_debt(root, measured, ceiling, parser=parser)
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 2 and "New undocumented" in result.stdout
    assert ceiling.read_bytes() == original


def test_source_removal_is_not_documentation_progress(parser: NativeParser, tmp_path: Path) -> None:
    """A removed original file cannot disappear from the protected native source cohort."""
    root, ceiling = _baseline(
        tmp_path / "repo",
        parser,
        "//! Native crate.\n/// Original module.\npub mod other;\n",
        extra={"other.rs": "pub fn old() {}\n"},
    )
    _git(root, "rm", "--", "src/other.rs")
    (root / "src/lib.rs").write_text("//! Native crate.\n/// Original function.\npub fn old() {}\n")
    result = _cli(root, ceiling)
    assert result.returncode == 2 and "cohort must not shrink" in result.stdout


def test_update_and_reintroduced_debt(parser: NativeParser, tmp_path: Path) -> None:
    """An actual v2 update lowers individual allowances and refuses restored documentation debt."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    (root / "src/lib.rs").write_text(OLD.replace("pub fn", "/// Original function.\npub fn"))
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(ceiling.read_text())
    assert data["schema_version"] == INDIVIDUAL_SCHEMA
    assert data["undocumented"] == 0 and data["provenance"]["debt_cases"] == []
    (root / "src/lib.rs").write_text(OLD)
    result = _cli(root, ceiling)
    assert result.returncode == 2 and "New undocumented" in result.stdout


@pytest.mark.parametrize("config", ["allow", "cap", "color"])
def test_actual_public_cli_cannot_hide_new_debt(
    parser: NativeParser, tmp_path: Path, config: str
) -> None:
    """The real ratchet rejects new compiler debt under attributes, cap-lints and color."""
    root, ceiling = _baseline(
        tmp_path / "repo", parser, "//! Native crate.\n/// Original function.\npub fn old() {}\n"
    )
    if config == "allow":
        source = "//! Native crate.\n#[allow(missing_docs)]\npub fn old() {}\n"
    else:
        source = OLD
        (root / ".cargo").mkdir()
        (root / ".cargo/config.toml").write_text(
            '[build]\nrustflags = ["--cap-lints", "allow"]\n'
            if config == "cap"
            else '[term]\ncolor = "always"\n'
        )
    (root / "src/lib.rs").write_text(source)
    result = _cli(root, ceiling)
    assert result.returncode == 2 and "New undocumented" in result.stdout


@pytest.mark.parametrize("fault", ["inflated", "original_scalar", "version", "manifest"])
def test_original_scope_and_producer_must_reproduce(
    parser: NativeParser, tmp_path: Path, fault: str
) -> None:
    """Refuse original ceiling/version mismatches and switching to an empty alternate crate."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    data = json.loads(ceiling.read_text())
    manifest = "Cargo.toml"
    if fault == "inflated":
        data["undocumented"] = 2
    elif fault == "original_scalar":
        data["undocumented"] = 0
    elif fault == "version":
        data["provenance"]["rustc"] = "unqualified compiler"
    elif fault == "manifest":
        other = root / "other"
        rust_project(
            other, "//! Alternate crate.\n/// Alternate function.\npub fn alternate() {}\n"
        )
        manifest = "other/Cargo.toml"
    ceiling.write_text(json.dumps(data))
    if fault in {"original_scalar", "version"}:
        _git(root, "add", "--", "ceiling.json")
        _git(
            root,
            "-c",
            "user.name=Native contract",
            "-c",
            "user.email=native@example.invalid",
            "commit",
            "-qm",
            "original record under test",
        )
    result = _cli(root, ceiling, manifest=manifest)
    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize(
    "fault",
    [
        "json",
        "object",
        "bool",
        "negative",
        "schema",
        "cases",
        "shape",
        "duplicate",
        "count",
        "new_allowance",
    ],
)
def test_invalid_candidate_cases_are_refused(
    parser: NativeParser, tmp_path: Path, fault: str
) -> None:
    """Malformed or invented candidate allowances cannot replace actual original compiler debt."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    assert _cli(root, ceiling, update=True).returncode == 0
    data = json.loads(ceiling.read_text())
    cases = data["provenance"]["debt_cases"]
    if fault == "bool":
        data["undocumented"] = True
    elif fault == "negative":
        data["undocumented"] = -1
    elif fault == "schema":
        data["schema_version"] = "unsupported"
    elif fault == "cases":
        data["provenance"]["debt_cases"] = None
    elif fault == "shape":
        del cases[0]["identity"]
    elif fault == "duplicate":
        cases.append(cases[0])
    elif fault == "count":
        data["undocumented"] = 0
    elif fault == "new_allowance":
        cases[0]["identity"] = "/function:invented"
    ceiling.write_text("{" if fault == "json" else "[]" if fault == "object" else json.dumps(data))
    result = _cli(root, ceiling)
    assert result.returncode == 2, result.stdout + result.stderr


def test_declaration_removal_is_not_documentation_repayment(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Removing an original undocumented function cannot silently repay retained ABI debt."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    original = ceiling.read_bytes()
    (root / "src/lib.rs").write_text("//! Native original crate.\n")
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 0
    with pytest.raises(RustMeasurementError, match="declarations must remain"):
        protect_rust_debt(root, measured, ceiling, parser=parser)
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 2 and "declarations must remain" in result.stdout
    assert ceiling.read_bytes() == original


@pytest.mark.parametrize("fault", ["downgrade", "original_cases"])
def test_original_individual_protocol_is_irreversible(
    parser: NativeParser, tmp_path: Path, fault: str
) -> None:
    """Original Git case identities must reproduce and cannot revert to scalar allowances."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    assert _cli(root, ceiling, update=True).returncode == 0
    original = json.loads(ceiling.read_text())
    if fault == "original_cases":
        original["provenance"]["debt_cases"][0]["identity"] = "/function:invented"
        ceiling.write_text(json.dumps(original))
    _git(root, "add", "--", "ceiling.json")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original individual protocol under test",
    )
    if fault == "downgrade":
        ceiling.write_text(json.dumps({"schema_version": LEGACY_SCHEMA, "undocumented": 1}))
    retained = ceiling.read_bytes()
    result = _cli(root, ceiling, update=True)
    expected = "cannot revert" if fault == "downgrade" else "cases do not reproduce"
    assert result.returncode == 2 and expected in result.stdout
    assert ceiling.read_bytes() == retained


def test_missing_candidate_needs_explicit_native_initialization(
    parser: NativeParser, tmp_path: Path
) -> None:
    """A missing candidate stays refused unless explicitly created from actual original debt."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    ceiling.unlink()
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    with pytest.raises(RustMeasurementError, match="No Rust documentation ceiling"):
        protect_rust_debt(root, measured, ceiling, parser=parser)
    assert not ceiling.exists()
    admitted = protect_rust_debt(root, measured, ceiling, allow_create=True, parser=parser)
    assert admitted.original.cases == measured.cases and not ceiling.exists()
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(ceiling.read_text())
    assert document["undocumented"] == 1 and document["schema_version"] == INDIVIDUAL_SCHEMA


def test_original_git_symlink_cannot_become_regular_source(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Refuse an actual original Git symlink even when the current path is regular Rust."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    alias = root / "src/alias.rs"
    alias.symlink_to("lib.rs")
    _git(root, "add", "--", "src/alias.rs")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original symlink under test",
    )
    alias.unlink()
    alias.write_text("/// Actual regular source.\npub fn alias() {}\n")
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    with pytest.raises(RustMeasurementError, match="Original Rust build sources"):
        protect_rust_debt(root, measured, ceiling, parser=parser)


def test_script_entry_preserves_ceiling_after_actual_compile_failure(
    parser: NativeParser,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The actual script entry propagates native compiler refusal without rewriting its ceiling."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    retained = ceiling.read_bytes()
    (root / "src/lib.rs").write_text("//! Native crate.\npub fn old() { absent(); }\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tools.rust_doc_ratchet",
            "--repo",
            str(root),
            "--manifest",
            "Cargo.toml",
            "--ceiling",
            str(ceiling),
            "--update",
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(REPO / "tools/rust_doc_ratchet.py"), run_name="__main__")
    assert result.value.code == 2 and "measurement failed" in capsys.readouterr().out
    assert ceiling.read_bytes() == retained


def test_original_engine_ceiling_cannot_be_bypassed_with_an_alternate_candidate(
    parser: NativeParser, tmp_path: Path
) -> None:
    """An external candidate cannot inflate the actual original engine allowance through the CLI."""
    root, ceiling = _baseline(tmp_path / "repo", parser)
    engine = root / "engine"
    engine.mkdir()
    for name in ("Cargo.toml", "Cargo.lock", "src"):
        (root / name).rename(engine / name)
    (root / "src").mkdir()
    (root / "src/lib.rs").write_bytes((engine / "src/lib.rs").read_bytes())
    original_ceiling = engine / "missing_docs_ceiling.json"
    original_ceiling.write_bytes(ceiling.read_bytes())
    _git(root, "add", "--all", "--", ".")
    _git(
        root,
        "-c",
        "user.name=Native contract",
        "-c",
        "user.email=native@example.invalid",
        "commit",
        "-qm",
        "original native engine allowance",
    )
    candidate = tmp_path / "alternate_ceiling.json"
    candidate.write_text(json.dumps({"undocumented": 100}))
    retained = candidate.read_bytes()
    actual = measure_rust_findings(root, "engine/Cargo.toml", parser=parser)
    assert actual.undocumented == 1
    assert {case.source for case in actual.cases} == {"engine/src/lib.rs"}
    with pytest.raises(RustMeasurementError, match="must not exceed"):
        protect_rust_debt(root, actual, candidate, parser=parser)
    result = _cli(root, candidate, update=True, manifest="engine/Cargo.toml")
    assert result.returncode == 2 and "must not exceed" in result.stdout
    assert (
        candidate.read_bytes() == retained and original_ceiling.read_bytes() == ceiling.read_bytes()
    )
