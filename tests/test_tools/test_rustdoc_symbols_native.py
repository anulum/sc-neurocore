# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — actual native Rust declaration identity contracts

"""Exercise Rust syntax identities, exact bytes and refusal through native CLI."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.rust_doc_symbols import NativeParser, RustSymbolError, build_parser

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def parser() -> NativeParser:
    """Build the real locked Rust parser once for this dedicated native surface."""
    return build_parser(REPO)


def test_qualified_members_and_unicode(parser: NativeParser, tmp_path: Path) -> None:
    """Resolve same-name fields and methods under their actual Unicode type owners."""
    source = """//! Exact-byte source.
pub struct Prvý { pub value: u8 }
pub struct Second { pub value: u8 }
impl Prvý { pub fn record(&self) {} }
impl Second { pub fn record(&self) {} }
pub mod inner { pub fn record() {} }
"""
    (tmp_path / "lib.rs").write_text(source)
    response = parser.read(tmp_path, ["lib.rs"])["lib.rs"]
    assert response.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    identities = {symbol.identity for symbol in response.symbols}
    assert {
        "/struct:Prvý/field:value",
        "/struct:Second/field:value",
        "/impl:Prvý/method:record",
        "/impl:Second/method:record",
        "/module:inner/function:record",
    } <= identities
    for symbol in response.symbols:
        assert source.encode()[symbol.start : symbol.end].decode() == symbol.name


def test_item_variants(parser: NativeParser, tmp_path: Path) -> None:
    """Collect nested enums, traits, foreign items and tuple fields from real syntax."""
    source = """
pub enum Choice { One { value: u8 }, Two(u8) }
pub struct Tuple(pub u8);
pub union Both { pub left: u8, pub right: u16 }
pub trait Action { const LIMIT: u8; type Output; fn execute(&self); }
pub struct Worker;
impl Action for Worker { const LIMIT: u8 = 1; type Output = u8; fn execute(&self) {} }
pub const LIMIT: u8 = 1;
pub static VALUE: u8 = 2;
pub type Alias = u8;
unsafe extern "C" { pub fn call(); pub static CURRENT: u8; pub type External; }
extern crate core;
macro_rules! exported { () => {} }
use core::mem;
"""
    (tmp_path / "lib.rs").write_text(source)
    symbols = parser.read(tmp_path, ["lib.rs"])["lib.rs"].symbols
    identities = {symbol.identity for symbol in symbols}
    assert {
        "/enum:Choice/variant:One/field:value",
        "/enum:Choice/variant:Two/field:0",
        "/struct:Tuple/field:0",
        "/union:Both/field:left",
        "/union:Both/field:right",
        "/trait:Action/constant:LIMIT",
        "/trait:Action/type:Output",
        "/trait:Action/method:execute",
        "/impl:Action for Worker/constant:LIMIT",
        "/impl:Action for Worker/type:Output",
        "/impl:Action for Worker/method:execute",
        "/constant:LIMIT",
        "/static:VALUE",
        "/type:Alias",
        "/extern:C/function:call",
        "/extern:C/static:CURRENT",
        "/extern:C/type:External",
        "/extern_crate:core",
        "/macro:exported",
    } <= identities
    assert len(symbols) == len(identities)


def test_source_cohort_and_line_shift(parser: NativeParser, tmp_path: Path) -> None:
    """Keep declaration identity stable after moving lines while rehashing actual bytes."""
    (tmp_path / "lib.rs").write_text("pub fn entry() {}\n")
    (tmp_path / "other.rs").write_text("pub fn entry() {}\n")
    before = parser.read(tmp_path, ["lib.rs", "other.rs"])
    (tmp_path / "lib.rs").write_text("\n\n/// Description.\npub fn entry() {}\n")
    after = parser.read(tmp_path, ["lib.rs", "other.rs"])
    assert set(after) == {"lib.rs", "other.rs"}
    assert before["lib.rs"].symbols[0].identity == after["lib.rs"].symbols[0].identity
    assert before["lib.rs"].source_sha256 != after["lib.rs"].source_sha256
    assert before["other.rs"] == after["other.rs"]


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["lib.rs", "lib.rs"],
        ["../lib.rs"],
        ["/lib.rs"],
        ["./lib.rs"],
        ["lib.rs/../lib.rs"],
        [" lib.rs"],
        ["lib.rs "],
        ["lib\\rs"],
        ["lib\n.rs"],
        ["lib\x7f.rs"],
        ["lib.txt"],
        ["absent.rs"],
        ["nested//lib.rs"],
    ],
)
def test_adapter_refuses_ambiguous_inputs(
    parser: NativeParser,
    tmp_path: Path,
    paths: list[str],
) -> None:
    """Reject absent, duplicated, escaping and lexically ambiguous real paths."""
    (tmp_path / "lib.rs").write_text("pub fn entry() {}")
    with pytest.raises(RustSymbolError):
        parser.read(tmp_path, paths)


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "null",
        "[1]",
        "[]",
        '["lib.rs", "lib.rs"]',
        '["../lib.rs"]',
        '["./lib.rs"]',
        '["absent.rs"]',
        '["lib.txt"]',
        '["nested//lib.rs"]',
    ],
)
def test_native_cli_refuses_invalid_protocol(
    parser: NativeParser,
    tmp_path: Path,
    raw: str,
) -> None:
    """Refuse invalid input in the actual compiled CLI without a partial response."""
    (tmp_path / "lib.rs").write_text("pub fn entry() {}")
    result = subprocess.run(
        [str(parser.executable)],
        input=raw,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("rustdoc_symbols:")


@pytest.mark.parametrize("source", ["pub fn {", "\udcff"])
def test_actual_parser_refuses_invalid_source(
    parser: NativeParser,
    tmp_path: Path,
    source: str,
) -> None:
    """Reject malformed Rust or UTF-8 in the real source parser, retaining nonzero status."""
    (tmp_path / "lib.rs").write_bytes(source.encode(errors="surrogateescape"))
    with pytest.raises(RustSymbolError, match="command failed"):
        parser.read(tmp_path, ["lib.rs"])


def test_symlink_refusal(parser: NativeParser, tmp_path: Path) -> None:
    """Reject symlink source access at both the adapter and compiled CLI boundary."""
    (tmp_path / "real.rs").write_text("pub fn entry() {}")
    (tmp_path / "alias.rs").symlink_to("real.rs")
    with pytest.raises(RustSymbolError):
        parser.read(tmp_path, ["alias.rs"])
    result = subprocess.run(
        [str(parser.executable)],
        input=json.dumps(["alias.rs"]),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2 and not result.stdout


def test_producer_change_is_refused(tmp_path: Path) -> None:
    """Build actual producer sources, then refuse a source rewrite before execution."""
    shutil.copytree(REPO / "tools/rustdoc_symbols", tmp_path / "tools/rustdoc_symbols")
    native = build_parser(tmp_path)
    (tmp_path / "lib.rs").write_text("pub fn entry() {}")
    source = tmp_path / "tools/rustdoc_symbols/src/main.rs"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(RustSymbolError, match="producer sources changed"):
        native.read(tmp_path, ["lib.rs"])


def test_missing_producer_refusal(tmp_path: Path) -> None:
    """Require the actual lockfile and producer sources before invoking Cargo."""
    with pytest.raises(RustSymbolError, match="producer inputs"):
        build_parser(tmp_path)


@pytest.mark.parametrize("change", ["rewrite", "remove"])
def test_bound_native_binary_cannot_drift(
    parser: NativeParser, tmp_path: Path, change: str
) -> None:
    """Reject a changed or absent copy of the actual Cargo-built binary before accepting output."""
    executable = tmp_path / "actual-native-parser"
    shutil.copy2(parser.executable, executable)
    observed = replace(parser, executable=executable)
    observed.verify()
    if change == "remove":
        executable.unlink()
    else:
        executable.write_bytes(executable.read_bytes() + b"changed")
    with pytest.raises(RustSymbolError):
        observed.verify()


def test_source_identity_separates_actual_cargo_outputs(tmp_path: Path) -> None:
    """Different real producer bytes cannot share Cargo's final executable pathname."""
    roots = [tmp_path / "original", tmp_path / "changed"]
    for root in roots:
        shutil.copytree(REPO / "tools/rustdoc_symbols", root / "tools/rustdoc_symbols")
    changed = roots[1] / "tools/rustdoc_symbols/src/main.rs"
    changed.write_text(changed.read_text() + "\n// Different observed producer bytes.\n")
    first, second = (build_parser(root) for root in roots)
    assert first.source_pins != second.source_pins
    assert first.executable != second.executable
    first.verify()
    second.verify()
    (tmp_path / "lib.rs").write_text("pub fn entry() {}")
    assert first.read(tmp_path, ["lib.rs"]) == second.read(tmp_path, ["lib.rs"])


def test_actual_empty_syntax_keeps_source_in_native_cohort(
    parser: NativeParser, tmp_path: Path
) -> None:
    """A source without declarations retains its native byte hash and empty symbol array."""
    (tmp_path / "lib.rs").write_text("//! A documented empty crate.\n")
    observed = parser.read(tmp_path, ["lib.rs"])
    assert set(observed) == {"lib.rs"} and observed["lib.rs"].symbols == ()


def test_missing_native_cargo_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An actual empty command path cannot create a qualified native producer."""
    shutil.copytree(REPO / "tools/rustdoc_symbols", tmp_path / "tools/rustdoc_symbols")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RustSymbolError, match="could not complete"):
        build_parser(tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "json",
        "object",
        "schema",
        "cohort",
        "hash",
        "array",
        "shape",
        "identity",
        "span",
        "past_end",
        "utf8_boundary",
        "duplicate",
        "source_change",
        "source_remove",
    ],
)
def test_real_compiled_protocol_faults_are_refused(tmp_path: Path, fault: str) -> None:
    """Compile controlled faults into the real parser after it parses the actual source.

    The native Cargo build, dependency lock, Rust AST traversal and source
    hashing execute normally. Only this private producer's published protocol
    is faulted; no command result or executable is substituted.
    """
    shutil.copytree(REPO / "tools/rustdoc_symbols", tmp_path / "tools/rustdoc_symbols")
    path = tmp_path / "tools/rustdoc_symbols/src/main.rs"
    source = path.read_text()
    original = """    let mut output = io::stdout().lock();
    serde_json::to_writer(
        &mut output,
        &Report {
            schema_version: "sc-neurocore.rustdoc-symbols.v1",
            sources,
        },
    )?;
    std::io::Write::flush(&mut output)?;"""
    assert source.count(original) == 1
    mutations = {
        "json": 'std::io::Write::write_all(&mut io::stdout().lock(), b"{")?;',
        "object": "wire = serde_json::Value::Null;",
        "schema": 'wire["schema_version"] = serde_json::json!("unsupported");',
        "cohort": 'wire["sources"] = serde_json::json!({});',
        "hash": 'wire["sources"]["lib.rs"]["source_sha256"] = serde_json::json!("0".repeat(64));',
        "array": 'wire["sources"]["lib.rs"]["symbols"] = serde_json::Value::Null;',
        "shape": 'wire["sources"]["lib.rs"]["symbols"][0]["extra"] = serde_json::json!(true);',
        "identity": 'wire["sources"]["lib.rs"]["symbols"][0]["identity"] = serde_json::json!("/wrong");',
        "span": 'wire["sources"]["lib.rs"]["symbols"][0]["end"] = serde_json::json!(0);',
        "past_end": 'wire["sources"]["lib.rs"]["symbols"][0]["end"] = serde_json::json!(fs::read("lib.rs")?.len() + 1);',
        "utf8_boundary": 'let start = wire["sources"]["lib.rs"]["symbols"][0]["start"].as_u64().unwrap(); wire["sources"]["lib.rs"]["symbols"][0]["start"] = serde_json::json!(start + 4);',
        "duplicate": 'let duplicate = wire["sources"]["lib.rs"]["symbols"][0].clone(); wire["sources"]["lib.rs"]["symbols"].as_array_mut().unwrap().push(duplicate);',
        "source_change": 'fs::write("lib.rs", "pub fn changed() {}")?;',
        "source_remove": 'fs::remove_file("lib.rs")?;',
    }
    replacement = """    let mut wire = serde_json::to_value(Report {
        schema_version: "sc-neurocore.rustdoc-symbols.v1",
        sources,
    })?;
    """ + mutations[fault]
    if fault != "json":
        replacement += "\n    serde_json::to_writer(io::stdout().lock(), &wire)?;"
    path.write_text(source.replace(original, replacement))
    native = build_parser(tmp_path)
    (tmp_path / "lib.rs").write_text(
        "pub fn prvý() {}" if fault == "utf8_boundary" else "pub fn entry() {}"
    )
    with pytest.raises(RustSymbolError):
        native.read(tmp_path, ["lib.rs"])


def test_native_output_failure_is_reported(parser: NativeParser, tmp_path: Path) -> None:
    """The real CLI must report an actual buffered stdout write failure as non-success."""
    (tmp_path / "lib.rs").write_text("pub fn entry() {}")
    read_end, write_end = os.pipe()
    os.close(read_end)
    with os.fdopen(write_end, "wb") as output:
        result = subprocess.run(
            [str(parser.executable)],
            cwd=tmp_path,
            input='["lib.rs"]',
            text=True,
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    assert result.returncode == 2 and result.stderr.startswith("rustdoc_symbols:")


def test_actual_cargo_alternate_binary_is_not_the_parser(tmp_path: Path) -> None:
    """Refuse a successful real Cargo build whose binary has another target identity."""
    tool = tmp_path / "tools/rustdoc_symbols"
    shutil.copytree(REPO / "tools/rustdoc_symbols", tool)
    manifest = tool / "Cargo.toml"
    text = manifest.read_text()
    assert text.count("publish = false") == 1
    manifest.write_text(
        text.replace("publish = false", "publish = false\nautobins = false")
        + '\n[[bin]]\nname = "alternate_native_binary"\npath = "src/main.rs"\n'
    )
    with pytest.raises(RustSymbolError, match="one native parser executable"):
        build_parser(tmp_path)


@pytest.mark.parametrize(
    "stdout",
    [
        "output-from-real-parser-procedural-macro",
        '{"reason":"build-finished","success":false}',
        '{"reason":"build-finished","success":true}',
    ],
)
def test_actual_macro_stdout_cannot_supply_parser_build_proof(tmp_path: Path, stdout: str) -> None:
    """Refuse real compiler macro output that corrupts or impersonates Cargo completion.

    Cargo and rustc execute normally with the native parser's locked dependencies.
    A private local procedural macro writes actual compiler stdout; no command,
    executable or returned transport is replaced.
    """
    tool = tmp_path / "tools/rustdoc_symbols"
    shutil.copytree(REPO / "tools/rustdoc_symbols", tool)
    macro = tool / "transport_fault"
    (macro / "src").mkdir(parents=True)
    (macro / "Cargo.toml").write_text(
        '[package]\nname = "transport_fault"\nversion = "0.0.0"\nedition = "2021"\n'
        + "[lib]\nproc-macro = true\n"
    )
    (macro / "src/lib.rs").write_text(
        "extern crate proc_macro;\n#[proc_macro_attribute]\n"
        "pub fn emit(_: proc_macro::TokenStream, item: proc_macro::TokenStream) "
        '-> proc_macro::TokenStream { println!("{}", ' + json.dumps(stdout) + "); item }\n"
    )
    manifest = tool / "Cargo.toml"
    manifest.write_text(manifest.read_text() + 'transport_fault = { path = "transport_fault" }\n')
    main = tool / "src/main.rs"
    text = main.read_text()
    assert text.count("fn main() {") == 1
    main.write_text(text.replace("fn main() {", "#[transport_fault::emit]\nfn main() {"))
    subprocess.run(
        ["cargo", "generate-lockfile", "--offline", "--manifest-path", str(manifest)],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=30,
    )
    with pytest.raises(RustSymbolError, match="Cargo"):
        build_parser(tmp_path)
