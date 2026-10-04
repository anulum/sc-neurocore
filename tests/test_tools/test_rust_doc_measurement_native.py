# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — actual Rust compiler documentation measurement contracts

"""Verify actual Cargo diagnostics, qualified declaration identities and source pins."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import shutil
import subprocess
import sys

import pytest

from tools.rust_doc_measurement import RustMeasurementError, measure_rust_findings
from tools.rust_doc_symbols import NativeParser, build_parser
from tools.doc_debt_ceiling import RatchetError
from tools.rust_doc_ratchet import measure

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parser() -> NativeParser:
    """Build the actual Rust source identity producer used with real Cargo findings."""
    return build_parser(REPO)


def rust_project(
    root: Path, source: str, *, crate_type: str = "lib", extra: dict[str, str] | None = None
) -> Path:
    """Create a real Git-owned Cargo library and generate its actual native lockfile."""
    root.mkdir()
    (root / "src").mkdir()
    (root / ".gitignore").write_text("native-target/\n")
    (root / "Cargo.toml").write_text(
        '[package]\nname = "native_debt"\nversion = "0.0.0"\nedition = "2021"\n'
        + '[lib]\ncrate-type = ["'
        + crate_type
        + '"]\n',
    )
    (root / "src/lib.rs").write_text(source)
    for name, contents in (extra or {}).items():
        (root / "src" / name).write_text(contents)
    commands = [
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
    ]
    for argv in commands:
        subprocess.run(argv, cwd=root, capture_output=True, check=True, timeout=30)
    return root


def test_actual_compiler_same_names_have_different_owners(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Six native diagnostics identify separate same-name methods and fields."""
    root = rust_project(
        tmp_path / "repo",
        """//! Native declaration ownership.
pub struct First { pub value: u8 }
pub struct Second { pub value: u8 }
impl First { pub fn record(&self) {} }
impl Second { pub fn record(&self) {} }
""",
    )
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 6 and measured.files == 1
    assert {case.identity for case in measured.cases} == {
        "/struct:First",
        "/struct:Second",
        "/struct:First/field:value",
        "/struct:Second/field:value",
        "/impl:First/method:record",
        "/impl:Second/method:record",
    }
    measured.verify(root)
    record = measured.to_public_dict()
    assert record["undocumented"] == 6 and record["source_pins"] == measured.pins


@pytest.mark.parametrize("crate_type", ["lib", "rlib", "cdylib"])
def test_zero_requires_a_real_library_artifact(
    parser: NativeParser, tmp_path: Path, crate_type: str
) -> None:
    """Documented libraries of supported native crate types return qualified zero."""
    root = rust_project(
        tmp_path / "repo",
        "//! Native crate.\n/// Answer the caller.\npub fn answer() {}\n",
        crate_type=crate_type,
    )
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 0 and measured.files == 0
    assert measured.cases == frozenset()


@pytest.mark.parametrize("config", ["allow", "expect", "cap", "color"])
def test_lint_suppression_cannot_hide_native_debt(
    parser: NativeParser,
    tmp_path: Path,
    config: str,
) -> None:
    """Keep real missing-doc findings despite attributes, cap-lints or colored output."""
    attribute = {"allow": "#[allow(missing_docs)]\n", "expect": "#[expect(missing_docs)]\n"}.get(
        config, ""
    )
    root = rust_project(tmp_path / "repo", "//! Native crate.\n" + attribute + "pub fn old() {}\n")
    (root / ".cargo").mkdir()
    if config == "cap":
        (root / ".cargo/config.toml").write_text('[build]\nrustflags = ["--cap-lints", "allow"]\n')
    elif config == "color":
        (root / ".cargo/config.toml").write_text('[term]\ncolor = "always"\n')
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 1
    assert next(iter(measured.cases)).identity == "/function:old"
    assert "--force-warn" in measured.argv


def test_unicode_line_moves_and_cached_replay(parser: NativeParser, tmp_path: Path) -> None:
    """Resolve Unicode byte spans and retain debt when Cargo replays cached diagnostics."""
    root = rust_project(tmp_path / "repo", "//! Unicode crate.\npub fn pôvodný() {}\n")
    first = measure_rust_findings(root, "Cargo.toml", parser=parser)
    cached = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert first.cases == cached.cases and first.pins == cached.pins
    (root / "src/lib.rs").write_text("\n\n//! Unicode crate.\npub fn pôvodný() {}\n")
    moved = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert moved.cases == first.cases and moved.pins != first.pins


def test_source_and_config_changes_invalidate_observation(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Refuse stale measurements after actual source, cohort and configuration edits."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    (root / "src/new.rs").write_text("/// New function.\npub fn new() {}\n")
    with pytest.raises(RustMeasurementError, match="inputs changed"):
        measured.verify(root)
    (root / "src/lib.rs").write_text("//! Native crate.\nmod new;\npub fn old() {}\n")
    newer = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert "src/new.rs" in newer.paths
    (root / "Cargo.toml").write_text((root / "Cargo.toml").read_text() + "\n")
    with pytest.raises(RustMeasurementError, match="inputs changed"):
        newer.verify(root)


@pytest.mark.parametrize(
    "bad",
    ["source", "compile", "manifest", "escape", "no_git", "no_lock", "symlink", "config_symlink"],
)
def test_unavailable_native_inputs_never_become_zero(
    parser: NativeParser,
    tmp_path: Path,
    bad: str,
) -> None:
    """Refuse actual unavailable Git, source, lockfile, compiler and manifest inputs."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    manifest = "Cargo.toml"
    if bad == "source":
        (root / "src/lib.rs").write_text("pub fn {")
    elif bad == "compile":
        (root / "src/lib.rs").write_text("//! Native crate.\npub fn old() { absent(); }")
    elif bad == "manifest":
        manifest = "absent.toml"
    elif bad == "escape":
        manifest = "../repo/Cargo.toml"
    elif bad == "no_git":
        (root / ".git").rename(root / "detached-native-git")
    elif bad == "no_lock":
        (root / "Cargo.lock").rename(root / "original-native-lock")
        (root / "Cargo.toml").write_text(
            (root / "Cargo.toml").read_text() + '\n[dependencies]\nserde = "=1.0.228"\n'
        )
    elif bad == "symlink":
        (root / "alias.rs").symlink_to("src/lib.rs")
    elif bad == "config_symlink":
        (root / ".cargo").mkdir()
        (root / "original-config.toml").write_text("")
        (root / ".cargo/config.toml").symlink_to(root / "original-config.toml")
    with pytest.raises(RustMeasurementError):
        measure_rust_findings(root, manifest, parser=parser)


def test_crate_documentation_is_an_individual_case(parser: NativeParser, tmp_path: Path) -> None:
    """Resolve rustc's crate-level diagnostic without inventing a declaration span."""
    root = rust_project(tmp_path / "repo", "/// Documented function.\npub fn old() {}\n")
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert measured.undocumented == 1 and next(iter(measured.cases)).identity == "/crate"


def test_bound_toolchain_observation_cannot_be_replaced(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Reject stale compiler and Cargo versions against actual unchanged native inputs."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    measured = measure_rust_findings(root, "Cargo.toml", parser=parser)
    for stale in (
        replace(measured, rustc_version="obsolete observation"),
        replace(measured, cargo_version="obsolete observation"),
    ):
        with pytest.raises(RustMeasurementError, match="toolchain changed"):
            stale.verify(root)


def test_public_legacy_measurement_refuses_native_compile_failure(tmp_path: Path) -> None:
    """The public tuple API propagates a real compiler failure instead of reporting zero."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() { absent(); }")
    with pytest.raises(RatchetError, match="measurement failed"):
        measure(root, "Cargo.toml")


def test_git_subdirectory_cannot_replace_full_native_cohort(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Refuse a real Git subdirectory that would lose maintained Rust source roots."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    with pytest.raises(RustMeasurementError, match="exact Git repository root"):
        measure_rust_findings(root / "src", "Cargo.toml", parser=parser)


def test_binary_target_cannot_stand_in_for_library(parser: NativeParser, tmp_path: Path) -> None:
    """A real successful Cargo metadata result for a binary cannot qualify library debt."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    (root / "src/lib.rs").rename(root / "src/main.rs")
    subprocess.run(
        ["git", "add", "--all", "--", "src"], cwd=root, check=True, capture_output=True, timeout=30
    )
    (root / "Cargo.toml").write_text(
        '[package]\nname = "native_debt"\nversion = "0.0.0"\nedition = "2021"\n'
    )
    with pytest.raises(RustMeasurementError, match="exactly one selected library"):
        measure_rust_findings(root, "Cargo.toml", parser=parser)


def test_macro_generated_declaration_requires_qualified_identity(
    parser: NativeParser, tmp_path: Path
) -> None:
    """An actual rustc macro diagnostic cannot become debt without a native syntax identity."""
    root = rust_project(
        tmp_path / "repo",
        "//! Native crate.\nmacro_rules! generate { () => { pub fn generated() {} }; }\ngenerate!();\n",
    )
    with pytest.raises(RustMeasurementError, match="could not be uniquely resolved"):
        measure_rust_findings(root, "Cargo.toml", parser=parser)


def test_missing_compiler_cannot_qualify_a_measured_source(
    parser: NativeParser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real Git inventory and native AST cannot substitute for an unavailable compiler."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    git = shutil.which("git")
    assert git is not None
    commands = tmp_path / "native-git-only"
    commands.mkdir()
    (commands / "git").symlink_to(git)
    monkeypatch.setenv("PATH", str(commands))
    with pytest.raises(RustMeasurementError, match="could not complete"):
        measure_rust_findings(root, "Cargo.toml", parser=parser)


def test_workspace_manifest_cannot_replace_a_selected_library(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Real workspace metadata must identify the exact selected package before debt is accepted."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    member = root / "member"
    member.mkdir()
    (root / "Cargo.toml").rename(member / "Cargo.toml")
    (root / "src").rename(member / "src")
    (root / "Cargo.toml").write_text('[workspace]\nmembers = ["member"]\nresolver = "2"\n')
    subprocess.run(["git", "add", "--all"], cwd=root, check=True, capture_output=True, timeout=30)
    with pytest.raises(RustMeasurementError, match="no qualified library package"):
        measure_rust_findings(root, "Cargo.toml", parser=parser)


def test_workspace_member_library_resolves_sources_from_the_workspace_root(
    parser: NativeParser, tmp_path: Path
) -> None:
    """A selected member reports sources relative to the workspace root, as the engine does."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\npub fn old() {}\n")
    member = root / "member"
    member.mkdir()
    (root / "Cargo.toml").rename(member / "Cargo.toml")
    (root / "src").rename(member / "src")
    (root / "Cargo.toml").write_text('[workspace]\nmembers = ["member"]\nresolver = "2"\n')
    subprocess.run(["git", "add", "--all"], cwd=root, check=True, capture_output=True, timeout=30)
    measured = measure_rust_findings(root, "member/Cargo.toml", parser=parser)
    assert {(case.source, case.identity) for case in measured.cases} == {
        ("member/src/lib.rs", "/function:old")
    }
    measured.verify(root)


def test_actual_macro_stdout_cannot_pass_after_cache_fill(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Repeated real compiler output must stay refused rather than pass through a cached artifact."""
    root = rust_project(
        tmp_path / "repo", "//! Native crate.\n/// Answer a caller.\npub fn old() {}\n"
    )
    macro = root / "noisy"
    (macro / "src").mkdir(parents=True)
    (macro / "Cargo.toml").write_text(
        '[package]\nname="noisy"\nversion="0.0.0"\nedition="2021"\n[lib]\nproc-macro=true\n'
    )
    (macro / "src/lib.rs").write_text(
        "//! Actual native macro.\nextern crate proc_macro;\n"
        "use proc_macro::TokenStream;\n/// Emit a compiler stdout observation.\n"
        "#[proc_macro]\npub fn emit(_input:TokenStream)->TokenStream {"
        'println!("output-from-real-procedural-macro"); TokenStream::new() }\n'
    )
    (root / "Cargo.toml").write_text(
        (root / "Cargo.toml").read_text() + '\n[dependencies]\nnoisy={path="noisy"}\n'
    )
    (root / "src/lib.rs").write_text(
        "//! Native crate.\nnoisy::emit!();\n/// Answer a caller.\npub fn old() {}\n"
    )
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    for _ in range(2):
        with pytest.raises(RustMeasurementError, match="could not be qualified"):
            measure_rust_findings(root, "Cargo.toml", parser=parser)
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text(json.dumps({"undocumented": 0}))
    retained = ceiling.read_bytes()
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
            "--update",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 2 and "could not be qualified" in result.stdout
    assert ceiling.read_bytes() == retained


def _compiler_stdout_project(root: Path, wire: object) -> Path:
    """Compile a real local macro that writes caller-controlled Cargo transport bytes."""
    root = rust_project(root, "//! Native crate.\n/// Original function.\npub fn old() {}\n")
    macro = root / "transport"
    (macro / "src").mkdir(parents=True)
    (macro / "Cargo.toml").write_text(
        '[package]\nname="transport"\nversion="0.0.0"\nedition="2021"\n[lib]\nproc-macro=true\n'
    )
    (macro / "src/lib.rs").write_text(
        "extern crate proc_macro;\n#[proc_macro]\n"
        "pub fn emit(_: proc_macro::TokenStream) -> proc_macro::TokenStream {"
        'println!("{}", ' + json.dumps(json.dumps(wire)) + "); proc_macro::TokenStream::new() }\n"
    )
    (root / "Cargo.toml").write_text(
        (root / "Cargo.toml").read_text() + '\n[dependencies]\ntransport={path="transport"}\n'
    )
    (root / "src/lib.rs").write_text("//! Native crate.\ntransport::emit!();\npub fn old() {}\n")
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return root


@pytest.mark.parametrize(
    "fault",
    ["object", "completion", "fresh", "spans", "primary", "offsets", "category", "duplicate"],
)
def test_actual_macro_cargo_records_cannot_create_a_false_measurement(
    parser: NativeParser, tmp_path: Path, fault: str
) -> None:
    """Real compiler stdout corruption cannot supply accepted cases or rewrite a ceiling.

    A native procedural macro emits the adverse record during actual library
    compilation. Cargo/rustc/Git and the source parser execute normally; their
    executables, subprocess results and adapter internals are not replaced.
    """
    root = tmp_path / "repo"
    source = "//! Native crate.\ntransport::emit!();\npub fn old() {}\n"
    start = source.index("old")
    span: dict[str, object] = {
        "is_primary": True,
        "file_name": "src/lib.rs",
        "byte_start": start,
        "byte_end": start + 3,
    }
    diagnostic: dict[str, object] = {
        "code": {"code": "missing_docs"},
        "message": "missing documentation for a function",
        "spans": [span],
    }
    target: dict[str, object] = {
        "name": "native_debt",
        "kind": ["lib"],
        "src_path": str(root / "src/lib.rs"),
    }
    wire: object = {
        "reason": "compiler-message",
        "manifest_path": str(root / "Cargo.toml"),
        "target": target,
        "message": diagnostic,
    }
    if fault == "object":
        wire = None
    elif fault == "completion":
        wire = {"reason": "build-finished", "success": False}
    elif fault == "fresh":
        wire = {
            "reason": "compiler-artifact",
            "manifest_path": str(root / "Cargo.toml"),
            "target": target,
            "fresh": True,
        }
    elif fault == "spans":
        diagnostic["spans"] = None
    elif fault == "primary":
        diagnostic["spans"] = []
    elif fault == "offsets":
        span["byte_start"] = True
    elif fault == "category":
        diagnostic["message"] = "unsupported native category"
    _compiler_stdout_project(root, wire)
    expected = {
        "object": "JSON objects",
        "completion": "successful Rust measurement",
        "fresh": "fresh selected library compilation",
        "spans": "require source spans",
        "primary": "require one primary source span",
        "offsets": "source offsets are invalid",
        "category": "category is unsupported",
        "duplicate": "identities are duplicated",
    }[fault]
    with pytest.raises(RustMeasurementError, match=expected):
        measure_rust_findings(root, "Cargo.toml", parser=parser)
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text(json.dumps({"undocumented": 0}))
    retained = ceiling.read_bytes()
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
            "--update",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 2 and ceiling.read_bytes() == retained


@pytest.mark.parametrize("location", ["outside", "ignored"])
def test_actual_compiler_source_outside_the_cohort_is_refused(
    parser: NativeParser, tmp_path: Path, location: str
) -> None:
    """Actual external or ignored module diagnostics cannot stand in for qualified source."""
    root = rust_project(tmp_path / "repo", "//! Native crate.\n")
    if location == "outside":
        (tmp_path / "external.rs").write_text("pub fn hidden() {}\n")
        path = "../../external.rs"
        expected = "escaped the source root"
    else:
        (root / "ignored.rs").write_text("pub fn hidden() {}\n")
        (root / ".gitignore").write_text((root / ".gitignore").read_text() + "ignored.rs\n")
        path = "../ignored.rs"
        expected = "outside the qualified cohort"
    (root / "src/lib.rs").write_text(
        f'//! Native crate.\n/// External module.\n#[path="{path}"]\npub mod external;\n'
    )
    with pytest.raises(RustMeasurementError, match=expected):
        measure_rust_findings(root, "Cargo.toml", parser=parser)


def test_actual_other_target_and_warning_do_not_invent_documentation_debt(
    parser: NativeParser, tmp_path: Path
) -> None:
    """Keep genuine build-script artifacts and deprecation warnings out of library debt."""
    root = rust_project(
        tmp_path / "repo",
        "//! Native crate.\n/// Older function.\n#[deprecated]\npub fn old() {}\n"
        "/// Documented caller.\npub fn caller() { old(); }\n",
    )
    (root / "build.rs").write_text("fn main() {}\n")
    observed = measure_rust_findings(root, "Cargo.toml", parser=parser)
    assert observed.undocumented == 0 and observed.cases == frozenset()
    assert "build.rs" in observed.paths
