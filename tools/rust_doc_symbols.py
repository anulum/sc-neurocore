# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — verified native Rust declaration identity adapter

"""Resolve Rust declaration identities through the actual pinned native parser."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from collections.abc import Sequence
from typing import cast

TOOL = "tools/rustdoc_symbols"
SCHEMA = "sc-neurocore.rustdoc-symbols.v1"
_INPUTS = ("Cargo.toml", "Cargo.lock", "src/main.rs", "src/items.rs")
_KINDS = {
    "module",
    "function",
    "struct",
    "enum",
    "union",
    "trait",
    "method",
    "field",
    "variant",
    "constant",
    "static",
    "type",
    "extern_crate",
    "macro",
}


class RustSymbolError(ValueError):
    """Refuse incomplete native execution, changed inputs or invalid identities."""


@dataclass(frozen=True)
class RustSymbol:
    """Associate an enclosing syntax identity with its identifier's byte range."""

    kind: str
    name: str
    identity: str
    start: int
    end: int


@dataclass(frozen=True)
class RustSourceSymbols:
    """Bind native syntax identities to the exact bytes submitted to syn."""

    source_sha256: str
    symbols: tuple[RustSymbol, ...]


@dataclass(frozen=True)
class NativeParser:
    """Retain a real Cargo-built parser and its observed source and binary hashes."""

    root: Path
    executable: Path
    source_pins: dict[str, str]
    executable_sha256: str
    cargo_version: str
    rustc_version: str

    def verify(self) -> None:
        """Require unchanged producer inputs and executable before accepting output."""
        if _producer_pins(self.root) != self.source_pins:
            raise RustSymbolError("Rust syntax producer sources changed during measurement.")
        if _hash(self.executable) != self.executable_sha256:
            raise RustSymbolError("Rust syntax producer executable changed during measurement.")

    def read(self, root: Path, paths: Sequence[str]) -> dict[str, RustSourceSymbols]:
        """Read a unique regular source cohort through the actual native executable.

        Parameters
        ----------
        root : Path
            Directory containing the exact repository-relative source paths.
        paths : sequence of str
            Rust files submitted as a JSON array, without symlink traversal.

        Returns
        -------
        dict of str to RustSourceSymbols
            Native declarations and hashes for every submitted source.

        Raises
        ------
        RustSymbolError
            If execution, input bytes, schema or byte ranges cannot be verified.
        """
        before = source_pins(root, paths)
        self.verify()
        raw = _run(root, [str(self.executable)], input_text=json.dumps(list(paths)))
        result = _decode(raw, before, root)
        self.verify()
        if source_pins(root, paths) != before:
            raise RustSymbolError("Rust syntax sources changed while the parser ran.")
        return result


def _hash(path: Path) -> str:
    """Hash observed regular producer or source bytes without substituting missing inputs."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RustSymbolError("A Rust syntax input or producer is unavailable.") from exc


def source_pins(root: Path, paths: Sequence[str]) -> dict[str, str]:
    """Hash normalized, unique, regular Rust source paths inside the physical root.

    Raises
    ------
    RustSymbolError
        If a path is ambiguous, missing, duplicated or traverses a symbolic link.
    """
    if not paths or len(set(paths)) != len(paths):
        raise RustSymbolError("Rust syntax sources must form a unique nonempty cohort.")
    physical = root.resolve()
    pins: dict[str, str] = {}
    for name in paths:
        path = PurePosixPath(name)
        absolute = physical / name
        if (
            not name
            or name.strip() != name
            or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not name.endswith(".rs")
            or absolute.resolve() != absolute
            or not absolute.is_file()
        ):
            raise RustSymbolError("Rust syntax inputs must be regular normalized relative paths.")
        pins[name] = _hash(absolute)
    return pins


def _producer_pins(root: Path) -> dict[str, str]:
    """Bind the parser source, manifest and exact dependency lockfile."""
    pins: dict[str, str] = {}
    for name in _INPUTS:
        path = root.resolve() / TOOL / name
        if path.resolve() != path or not path.is_file():
            raise RustSymbolError("Rust syntax producer inputs must be regular local files.")
        pins[name] = _hash(path)
    return pins


def _run(root: Path, argv: list[str], *, input_text: str | None = None) -> str:
    """Require actual native build or parser execution and successful terminal status."""
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustSymbolError("The native Rust syntax command could not complete.") from exc
    if completed.returncode != 0:
        raise RustSymbolError("The native Rust syntax command failed.")
    return completed.stdout


def _object(value: object) -> dict[str, object]:
    """Require object-valued native JSON records with string keys."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RustSymbolError("The native Rust syntax response requires JSON objects.")
    return cast(dict[str, object], value)


def _decode(raw: str, pins: dict[str, str], root: Path) -> dict[str, RustSourceSymbols]:
    """Validate source hashes and syntax identity shapes for the entire submitted cohort."""
    try:
        data = _object(json.loads(raw))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RustSymbolError("The native Rust syntax response must be valid JSON.") from exc
    if set(data) != {"schema_version", "sources"} or data["schema_version"] != SCHEMA:
        raise RustSymbolError("The native Rust syntax schema is unsupported.")
    sources = _object(data["sources"])
    if set(sources) != set(pins):
        raise RustSymbolError("The native Rust syntax response changed the submitted cohort.")
    result: dict[str, RustSourceSymbols] = {}
    for path, value in sources.items():
        entry = _object(value)
        if set(entry) != {"source_sha256", "symbols"} or entry["source_sha256"] != pins[path]:
            raise RustSymbolError("The native Rust syntax response parsed different source bytes.")
        try:
            source_bytes = (root.resolve() / path).read_bytes()
        except OSError as exc:
            raise RustSymbolError("The observed Rust syntax source bytes are unavailable.") from exc
        if hashlib.sha256(source_bytes).hexdigest() != pins[path]:
            raise RustSymbolError("Rust syntax sources changed while the parser ran.")
        raw_symbols = entry["symbols"]
        if not isinstance(raw_symbols, list):
            raise RustSymbolError("The native Rust syntax declarations must be an array.")
        symbols: list[RustSymbol] = []
        spans: set[tuple[int, int]] = set()
        for raw_symbol in raw_symbols:
            symbol = _object(raw_symbol)
            if set(symbol) != {"kind", "name", "identity", "start", "end"}:
                raise RustSymbolError("The native Rust syntax declaration shape is invalid.")
            kind, name, identity = (symbol[key] for key in ("kind", "name", "identity"))
            start, end = symbol["start"], symbol["end"]
            if (
                not isinstance(kind, str)
                or kind not in _KINDS
                or not isinstance(name, str)
                or not name
                or not isinstance(identity, str)
                or not identity.endswith(f"/{kind}:{name}")
                or isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or isinstance(end, bool)
                or not isinstance(end, int)
                or end <= start
                or end > len(source_bytes)
                or (start, end) in spans
            ):
                raise RustSymbolError("The native Rust syntax declaration identity is invalid.")
            try:
                source_bytes[start:end].decode("utf-8")
            except UnicodeError as exc:
                raise RustSymbolError(
                    "The native Rust syntax declaration splits a UTF-8 code point."
                ) from exc
            spans.add((start, end))
            symbols.append(RustSymbol(kind, name, identity, start, end))
        result[path] = RustSourceSymbols(pins[path], tuple(symbols))
    return result


def build_parser(root: Path | None = None) -> NativeParser:
    """Build the lock-pinned native parser and retain its actual Cargo artifact.

    An output directory bound to the producer root and source hashes separates
    builds whose shared binary filename could otherwise overwrite each other.
    A successful Cargo artifact identifies the producer executable. Source and
    executable observations bracket execution; they do not attest continuous
    filesystem immutability or the complete external Rust compiler installation.

    Parameters
    ----------
    root : Path or None
        Producer repository; defaults to this adapter's actual source checkout.

    Returns
    -------
    NativeParser
        Executable with unchanged source, lockfile and binary hashes.

    Raises
    ------
    RustSymbolError
        If Cargo fails or omits a unique executable for the exact manifest.
    """
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    pins = _producer_pins(root)
    manifest = root / TOOL / "Cargo.toml"
    digest = hashlib.sha256(
        json.dumps(
            {"root": str(root), "pins": pins}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    target_base = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "target")))
    output_dir = target_base / "rustdoc-symbols" / digest
    cargo_version = _run(root, ["cargo", "--version"]).strip()
    rustc_version = _run(root, ["rustc", "--version"]).strip()
    if not cargo_version.startswith("cargo ") or not rustc_version.startswith("rustc "):
        raise RustSymbolError("Native Rust syntax toolchain versions are unavailable.")
    raw = _run(
        root,
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(output_dir),
            "--message-format=json",
        ],
    )
    artifacts: list[Path] = []
    finished = False
    try:
        for line in raw.splitlines():
            message = _object(json.loads(line))
            if message.get("reason") == "build-finished":
                if finished or message.get("success") is not True:
                    raise RustSymbolError("Cargo did not finish one successful parser build.")
                finished = True
            if message.get("reason") == "compiler-artifact" and message.get("manifest_path") == str(
                manifest
            ):
                executable = message.get("executable")
                target = _object(message.get("target"))
                if (
                    target.get("name") == "sc_neurocore_rustdoc_symbols"
                    and target.get("kind") == ["bin"]
                    and isinstance(executable, str)
                ):
                    artifacts.append(Path(executable))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RustSymbolError("Cargo did not return valid parser build JSON.") from exc
    if not finished or len(artifacts) != 1:
        raise RustSymbolError("Cargo did not identify one native parser executable.")
    parser = NativeParser(
        root, artifacts[0], pins, _hash(artifacts[0]), cargo_version, rustc_version
    )
    parser.verify()
    return parser
