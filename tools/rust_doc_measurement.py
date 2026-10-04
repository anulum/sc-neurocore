# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native compiler Rust documentation findings

"""Bind compiler missing-doc diagnostics to qualified syntax and observed inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import cast
from uuid import uuid4

from tools.docstring_policy_git import git_bytes
from tools.docstring_policy_scope import PolicyError
from tools.rust_doc_symbols import (
    NativeParser,
    RustSourceSymbols,
    RustSymbolError,
    build_parser,
    source_pins,
)


class RustMeasurementError(ValueError):
    """Refuse incomplete compiler execution, ambiguous sources or unresolved debt."""


#: Identity prefix of an original macro-generated declaration without a syntax identity.
EXPANSION_PREFIX = "/expansion:"


@dataclass(frozen=True, order=True)
class RustDebtCase:
    """Identify missing documentation independently of warning formatting and lines."""

    source: str
    identity: str

    def to_public_dict(self) -> dict[str, str]:
        """Serialize the source and qualified declaration allowed by original debt."""
        return {"source": self.source, "identity": self.identity}


@dataclass(frozen=True)
class RustMeasurement:
    """Retain actual compiler cases, source cohort and bounded observed provenance."""

    manifest: str
    paths: tuple[str, ...]
    pins: dict[str, str]
    cases: frozenset[RustDebtCase]
    declarations: frozenset[RustDebtCase]
    rustc_version: str
    cargo_version: str
    argv: tuple[str, ...]
    producer_sha256: str

    @property
    def undocumented(self) -> int:
        """Return the number of uniquely resolved missing-doc compiler diagnostics."""
        return len(self.cases)

    @property
    def files(self) -> int:
        """Return the number of source files with missing-doc diagnostics."""
        return len({case.source for case in self.cases})

    def verify(self, root: Path) -> None:
        """Recheck local Git source scope, config bytes and native toolchain versions."""
        if (
            tuple(rust_scope(root)) != self.paths
            or capture_inputs(root, self.paths, self.manifest) != self.pins
        ):
            raise RustMeasurementError("Rust documentation inputs changed during measurement.")
        if (
            _run(root, ["rustc", "--version"]).strip() != self.rustc_version
            or _run(root, ["cargo", "--version"]).strip() != self.cargo_version
        ):
            raise RustMeasurementError(
                "The Rust documentation toolchain changed during measurement."
            )

    def to_public_dict(self) -> dict[str, object]:
        """Serialize actual compiler argv, source observations and declaration cases.

        The manifest's library configuration is measured. Other targets,
        platform cfgs, macro-generated declarations and external dependency
        inputs are not automatically qualified by this local observation.
        """
        return {
            "schema_version": "sc-neurocore.rust-doc-measurement.v1",
            "manifest": self.manifest,
            "paths": list(self.paths),
            "source_pins": dict(self.pins),
            "rustc": self.rustc_version,
            "cargo": self.cargo_version,
            "argv": list(self.argv),
            "undocumented": self.undocumented,
            "files": self.files,
            "debt_cases": [case.to_public_dict() for case in sorted(self.cases)],
            "declarations": [case.to_public_dict() for case in sorted(self.declarations)],
            "syntax_producer_sha256": self.producer_sha256,
            "qualification_scope": "fresh selected library compiler diagnostics and observed Git Rust/config inputs",
        }


def _run(root: Path, argv: list[str]) -> str:
    """Require terminal native compiler execution before reading structured output."""
    try:
        result = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, check=False, timeout=1800
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustMeasurementError(
            "The native cargo rustc measurement could not complete."
        ) from exc
    if result.returncode != 0:
        raise RustMeasurementError("The native cargo rustc measurement failed.")
    return result.stdout


def _object(value: object) -> dict[str, object]:
    """Refuse Cargo records whose decoded value is not an object."""
    if not isinstance(value, dict):
        raise RustMeasurementError("Native Cargo diagnostics require JSON objects.")
    return cast(dict[str, object], value)


def _names(root: Path) -> list[str]:
    """Read the exact native Git source and config inventory without environment redirection."""
    try:
        top = git_bytes(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        if Path(top).resolve() != root.resolve():
            raise RustMeasurementError("Rust documentation requires the exact Git repository root.")
        raw = git_bytes(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        if raw and not raw.endswith(b"\0"):
            raise RustMeasurementError("The Git Rust cohort is incomplete.")
        return sorted({name.decode("utf-8") for name in raw.split(b"\0") if name})
    except (PolicyError, UnicodeError) as exc:
        raise RustMeasurementError("The native Git Rust cohort could not be read.") from exc


def rust_scope(root: Path) -> list[str]:
    """Include tracked and nonignored untracked Rust inputs across repository roots."""
    paths = [name for name in _names(root) if name.endswith(".rs")]
    try:
        source_pins(root, paths)
    except RustSymbolError as exc:
        raise RustMeasurementError(
            "The Git Rust cohort contains unavailable or ambiguous inputs."
        ) from exc
    return paths


def capture_inputs(root: Path, paths: tuple[str, ...], manifest: str) -> dict[str, str]:
    """Hash the exact Rust cohort, selected manifest and present native build configs."""
    try:
        pins = source_pins(root, paths)
        configs = {
            name
            for name in _names(root)
            if Path(name).name
            in {"Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"}
            or "/.cargo/" in "/" + name
        } | {manifest}
        for name in sorted(configs):
            path = PurePosixPath(name)
            absolute = root.resolve() / name
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name
                or absolute.resolve() != absolute
                or not absolute.is_file()
            ):
                raise RustMeasurementError("Rust build inputs must be regular local files.")
            pins[name] = hashlib.sha256(absolute.read_bytes()).hexdigest()
        return pins
    except (OSError, RustSymbolError) as exc:
        raise RustMeasurementError("Rust documentation build inputs are unavailable.") from exc


def _is_library(target: dict[str, object]) -> bool:
    """Recognize native library crate kinds, including the engine cdylib and rlib."""
    kinds = target.get("kind")
    return isinstance(kinds, list) and bool(
        {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}.intersection(kinds)
    )


def _library(root: Path, manifest: str) -> tuple[str, str, Path]:
    """Resolve the selected manifest to its Cargo library name, source and workspace root."""
    data = _object(
        json.loads(
            _run(
                root,
                [
                    "cargo",
                    "metadata",
                    "--locked",
                    "--no-deps",
                    "--format-version=1",
                    "--manifest-path",
                    manifest,
                ],
            )
        )
    )
    packages = data.get("packages")
    workspace = data.get("workspace_root")
    if not isinstance(packages, list) or not isinstance(workspace, str) or not workspace:
        raise RustMeasurementError("Cargo did not identify the selected package.")
    for value in packages:
        package = _object(value)
        if package.get("manifest_path") != str(root.resolve() / manifest):
            continue
        targets = package.get("targets")
        if not isinstance(targets, list):
            raise RustMeasurementError("Cargo did not identify the selected library.")
        libraries = [
            _object(target)
            for target in targets
            if isinstance(target, dict) and _is_library(target)
        ]
        if (
            len(libraries) != 1
            or not isinstance(libraries[0].get("name"), str)
            or not isinstance(libraries[0].get("src_path"), str)
        ):
            raise RustMeasurementError("Cargo must identify exactly one selected library.")
        return (
            cast(str, libraries[0]["name"]),
            cast(str, libraries[0]["src_path"]),
            Path(workspace),
        )
    raise RustMeasurementError("The selected manifest has no qualified library package.")


def _span_text(span: dict[str, object]) -> str | None:
    """Join the exact source lines that the compiler attached to one span."""
    rows = span.get("text")
    if not isinstance(rows, list) or not rows:
        return None
    lines = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            return None
        lines.append(row["text"].strip())
    return "\n".join(lines)


def _expansion_identity(span: dict[str, object]) -> str | None:
    """Name a macro-generated declaration by its macro, call site and definition text.

    The syntax parser cannot see a declaration that exists only after macro
    expansion. The compiler reports the invoked macro and both source texts, so
    the identity survives line moves and changes with either text.
    """
    expansion = span.get("expansion")
    if not isinstance(expansion, dict):
        return None
    call, macro = expansion.get("span"), expansion.get("macro_decl_name")
    if not isinstance(call, dict) or not isinstance(macro, str) or not macro.strip("!"):
        return None
    definition, invocation = _span_text(span), _span_text(call)
    origin = call.get("file_name")
    if not definition or not invocation or not isinstance(origin, str):
        return None
    digest = hashlib.sha256(
        json.dumps([origin, invocation, definition], ensure_ascii=False).encode()
    ).hexdigest()
    return f"{EXPANSION_PREFIX}{macro.strip('!')}:{digest}"


def _case(
    root: Path,
    workspace_root: Path,
    diagnostic: dict[str, object],
    symbols: dict[str, RustSourceSymbols],
    *,
    admit_expansions: bool = False,
) -> RustDebtCase:
    """Resolve one real compiler primary span to a unique qualified syntax identity.

    Cargo starts the compiler in the workspace root, so a member library reports
    its sources relative to that root and a standalone package to its own directory.
    A macro-generated declaration has no syntax identity; it is refused unless the
    caller measures an original baseline and admits its expansion identity.
    """
    spans = diagnostic.get("spans")
    if not isinstance(spans, list):
        raise RustMeasurementError("Rust missing-doc diagnostics require source spans.")
    primary = [
        _object(span) for span in spans if isinstance(span, dict) and span.get("is_primary") is True
    ]
    if len(primary) != 1:
        raise RustMeasurementError("Rust missing-doc diagnostics require one primary source span.")
    span = primary[0]
    name, start, end = (span.get(key) for key in ("file_name", "byte_start", "byte_end"))
    if (
        not isinstance(name, str)
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise RustMeasurementError("Rust missing-doc diagnostic source offsets are invalid.")
    try:
        path = (workspace_root / name).resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RustMeasurementError("Rust compiler diagnostics escaped the source root.") from exc
    if path not in symbols:
        raise RustMeasurementError(
            "Rust compiler diagnostic source is outside the qualified cohort."
        )
    message = diagnostic.get("message")
    if message == "missing documentation for the crate":
        return RustDebtCase(path, "/crate")
    if not isinstance(message, str) or not message.startswith("missing documentation for "):
        raise RustMeasurementError("The Rust compiler missing-doc category is unsupported.")
    category = (
        message.removeprefix("missing documentation for ").removeprefix("a ").removeprefix("an ")
    )
    kind = {
        "associated function": "method",
        "associated constant": "constant",
        "associated type": "type",
        "struct field": "field",
        "type alias": "type",
        "enum variant": "variant",
    }.get(category, category)
    matches = [
        symbol
        for symbol in symbols[path].symbols
        if symbol.kind == kind and start <= symbol.start and symbol.end <= end
    ]
    if len(matches) != 1:
        expansion = _expansion_identity(span) if admit_expansions else None
        if expansion is None:
            raise RustMeasurementError(
                "The Rust compiler declaration identity could not be uniquely resolved."
            )
        return RustDebtCase(path, expansion)
    return RustDebtCase(path, matches[0].identity)


def measure_rust_findings(
    root: Path,
    manifest: str,
    *,
    parser: NativeParser | None = None,
    admit_expansions: bool = False,
) -> RustMeasurement:
    """Measure actual Cargo JSON diagnostics with unsuppressible missing-doc lint.

    Parameters
    ----------
    root : Path
        Exact Git repository root whose native Rust cohort is observed.
    manifest : str
        Selected repository-relative library manifest.
    parser : NativeParser or None
        Existing verified native syntax producer, or a fresh locked build.
    admit_expansions : bool
        Give undocumented macro-generated declarations an expansion identity
        instead of refusing them. Only an original baseline may use this.

    Returns
    -------
    RustMeasurement
        Uniquely resolved actual compiler cases and their local source pins.

    Raises
    ------
    RustMeasurementError
        If the compiler fails, provenance changes or diagnostics cannot resolve.
    """
    try:
        root = root.resolve()
        paths = tuple(rust_scope(root))
        pins = capture_inputs(root, paths, manifest)
        native = parser or build_parser()
        symbols = native.read(root, paths)
        rustc = _run(root, ["rustc", "--version"]).strip()
        cargo = _run(root, ["cargo", "--version"]).strip()
        target_name, target_source, workspace_root = _library(root, manifest)
        input_digest = hashlib.sha256(
            json.dumps(pins, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        argv = (
            "cargo",
            "rustc",
            "--locked",
            "--lib",
            "--manifest-path",
            manifest,
            "--message-format=json",
            "--",
            "--force-warn",
            "missing_docs",
            "--cfg",
            f'sc_neurocore_doc_measurement="{input_digest}"',
            "--cfg",
            f'sc_neurocore_doc_execution="{uuid4().hex}"',
        )
        raw = _run(root, list(argv))
        cases: set[RustDebtCase] = set()
        finished = artifact = False
        for line in raw.splitlines():
            message = _object(json.loads(line))
            reason = message.get("reason")
            if reason == "build-finished":
                if finished or message.get("success") is not True:
                    raise RustMeasurementError(
                        "Cargo did not finish one successful Rust measurement."
                    )
                finished = True
            if reason not in {"compiler-artifact", "compiler-message"}:
                continue
            if message.get("manifest_path") != str(root / manifest):
                continue
            target = _object(message.get("target"))
            if (
                target.get("name") != target_name
                or target.get("src_path") != target_source
                or not _is_library(target)
            ):
                continue
            if reason == "compiler-artifact":
                if artifact or message.get("fresh") is not False:
                    raise RustMeasurementError(
                        "Cargo did not prove one fresh selected library compilation."
                    )
                artifact = True
                continue
            diagnostic = _object(message.get("message"))
            code = diagnostic.get("code")
            if not isinstance(code, dict) or code.get("code") != "missing_docs":
                continue
            case = _case(
                root, workspace_root, diagnostic, symbols, admit_expansions=admit_expansions
            )
            if case in cases:
                raise RustMeasurementError(
                    "Rust missing-doc declaration identities are duplicated."
                )
            cases.add(case)
        if not finished or not artifact:
            raise RustMeasurementError(
                "Cargo omitted successful measurement or library artifact proof."
            )
        declarations = {RustDebtCase(path, "/crate") for path in symbols}
        declarations.update(
            RustDebtCase(path, symbol.identity)
            for path, source in symbols.items()
            for symbol in source.symbols
        )
        result = RustMeasurement(
            manifest,
            paths,
            pins,
            frozenset(cases),
            frozenset(declarations),
            rustc,
            cargo,
            argv,
            native.executable_sha256,
        )
        native.verify()
        result.verify(root)
        return result
    except (RustSymbolError, json.JSONDecodeError, UnicodeError) as exc:
        raise RustMeasurementError(
            "Native Rust documentation findings could not be qualified."
        ) from exc
