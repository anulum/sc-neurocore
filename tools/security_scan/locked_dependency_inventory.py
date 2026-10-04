# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Committed dependency audit inventory

"""Discover committed locks without installing or rewriting dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from tools.security_scan.python_dependency_profiles import discover_dependency_profiles

LOCK_KINDS = {
    "Cargo.lock": "rust",
    "package-lock.json": "npm",
    "go.mod": "go",
    "go.sum": "go-sum",
    "Manifest.toml": "julia",
    "pixi.lock": "pixi",
}


@dataclass(frozen=True)
class LockInput:
    """Bind one tracked lock input to its ecosystem and exact content digest."""

    path: str
    ecosystem: str
    sha256: str


def inventory_locks(repo_root: Path) -> tuple[LockInput, ...]:
    """Discover all tracked locks and reject unsupported or unreadable inputs.

    Parameters
    ----------
    repo_root : Path
        Git checkout containing the maintained dependency profiles.

    Returns
    -------
    tuple of LockInput
        Sorted locks, including Go checksum inputs and every Python profile.

    Raises
    ------
    OSError, ValueError, subprocess.SubprocessError
        Git discovery, lock custody or a maintained profile is unavailable.
        An additional lock format requires an audit implementation first.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, check=True, timeout=30
    )
    paths = set(result.stdout.decode("utf-8").split("\0")) - {""}
    profiles = discover_dependency_profiles(repo_root)
    python_paths = {profile.path for profile in profiles}
    if not python_paths.issubset(paths):
        raise ValueError("Python audit profiles must be committed inputs.")
    locks: list[LockInput] = []
    for path in sorted(paths):
        name = Path(path).name
        kind = "python" if path in python_paths else LOCK_KINDS.get(name)
        if kind is None:
            if name.endswith(".lock") or name in {"pnpm-lock.yaml", "yarn.lock", "pylock.toml"}:
                raise ValueError(f"Dependency lock has no audit implementation: {path}")
            continue
        absolute = repo_root / path
        if absolute.is_symlink() or not absolute.is_file():
            raise ValueError(f"Dependency lock must be a regular file: {path}")
        if kind == "go-sum" and str(Path(path).with_name("go.mod")) not in paths:
            raise ValueError("Go checksum input has no committed module manifest.")
        locks.append(LockInput(path, kind, hashlib.sha256(absolute.read_bytes()).hexdigest()))
    return tuple(locks)


def julia_queries(raw: bytes) -> tuple[dict[str, Any], ...]:
    """Extract exact Julia package versions for the OSV querybatch API.

    Parameters
    ----------
    raw : bytes
        Complete committed Julia Manifest.toml, with all dependency entries.

    Returns
    -------
    tuple of dict
        OSV package/version queries, retaining Julia build suffixes.

    Raises
    ------
    ValueError
        Manifest format, dependency names or exact versions are unavailable.
        Path, Git and unversioned entries need a separately reviewed identity.
    """
    manifest = tomllib.loads(raw.decode("utf-8"))
    dependencies = manifest.get("deps")
    if manifest.get("manifest_format") != "2.0" or not isinstance(dependencies, dict):
        raise ValueError("Julia audit requires a version-2 dependency manifest.")
    queries: list[dict[str, Any]] = []
    for name, entries in sorted(dependencies.items()):
        if not name or not isinstance(entries, list) or not entries:
            raise ValueError("Julia dependency entry is invalid.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Julia dependency identity is invalid.")
            version = entry.get("version")
            if (
                not isinstance(version, str)
                or not version
                or "path" in entry
                or "repo-url" in entry
            ):
                raise ValueError("Julia dependency needs an exact registry version.")
            queries.append({"package": {"ecosystem": "Julia", "name": name}, "version": version})
    if not queries:
        raise ValueError("Julia dependency manifest is empty.")
    return tuple(queries)


def go_queries(
    returncode: int, payload: object, checksum: str | None
) -> tuple[dict[str, Any], ...]:
    """Extract the Go standard library and every locked module version for OSV.

    Parameters
    ----------
    returncode : int
        Exit status of the real ``go mod edit -json`` manifest parser.
    payload : object
        JSON printed by that parser for one committed module manifest.
    checksum : str or None
        Complete committed go.sum beside the manifest, when one exists.

    Returns
    -------
    tuple of dict
        Sorted OSV package/version queries, including checksum-only versions.
        Call analysis is neither needed nor allowed to prune advisory findings.

    Raises
    ------
    ValueError
        The parser failed, a module is replaced or a version is not exact.
        Replaced modules need a separately reviewed advisory identity.
    """
    if returncode != 0 or not isinstance(payload, dict) or payload.get("Replace"):
        raise ValueError("Go audit requires readable, unreplaced module identities.")
    version = payload.get("Go")
    if not isinstance(version, str) or not version:
        raise ValueError("Go standard library version is unavailable.")
    identities = {("stdlib", version)}
    requirements = payload.get("Require") or []
    if not isinstance(requirements, list):
        raise ValueError("Go module requirements are invalid.")
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Go module requirement is invalid.")
        name, pinned = requirement.get("Path"), requirement.get("Version")
        if not isinstance(name, str) or not isinstance(pinned, str) or not pinned.startswith("v"):
            raise ValueError("Go module requires exact version identities.")
        identities.add((name, pinned[1:]))
    if checksum is not None:
        for line in checksum.splitlines():
            fields = line.split()
            if len(fields) != 3 or not fields[1].startswith("v") or not fields[2].startswith("h1:"):
                raise ValueError("Go checksum input is invalid.")
            identities.add((fields[0], fields[1][1:].removesuffix("/go.mod")))
    return tuple(
        {"package": {"ecosystem": "Go", "name": name}, "version": version}
        for name, version in sorted(identities)
    )


def pixi_package_count(raw: bytes) -> int:
    """Count the package records of one committed Pixi lock before its audit.

    Parameters
    ----------
    raw : bytes
        Complete committed pixi.lock, read before the scanner is started.

    Returns
    -------
    int
        Number of locked package records the audit report must account for.

    Raises
    ------
    ValueError, yaml.YAMLError
        The lock is not readable YAML or has no package record list.
        Such a lock cannot bound the coverage of any scanner report.
    """
    payload = yaml.safe_load(raw)
    packages = payload.get("packages") if isinstance(payload, dict) else None
    if not isinstance(packages, list):
        raise ValueError("Pixi lock package inventory is unavailable.")
    return len(packages)


def cargo_audit_configuration(repo_root: Path, cargo_home: Path) -> dict[str, str]:
    """Bind effective Cargo audit settings and require a fresh official database.

    Parameters
    ----------
    repo_root : Path
        Scanner working directory, whose .cargo/audit.toml takes precedence.
    cargo_home : Path
        Effective Cargo home, used only when project audit settings are absent.

    Returns
    -------
    dict of str
        Effective configuration path and content digest, or an empty mapping
        when the scanner uses its defaults. No configuration is rewritten.

    Raises
    ------
    OSError, ValueError
        Configuration custody is unavailable or database settings disable
        refresh, admit a stale database or override the reviewed source/cache.
    """
    configuration = repo_root / ".cargo/audit.toml"
    if not configuration.exists():
        configuration = cargo_home / "audit.toml"
    if not configuration.exists():
        return {}
    if configuration.is_symlink() or not configuration.is_file():
        raise ValueError("Cargo audit settings require a regular configuration input.")
    raw = configuration.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    database = payload.get("database", {})
    if not isinstance(database, dict):
        raise ValueError("Cargo audit database settings are invalid.")
    if database.get("fetch", True) is not True or database.get("stale", False) is not False:
        raise ValueError("Cargo audit requires database refresh and refuses stale acceptance.")
    if "path" in database or database.get("url", "https://github.com/RustSec/advisory-db") != (
        "https://github.com/RustSec/advisory-db"
    ):
        raise ValueError("Cargo audit database overrides require a reviewed custody adapter.")
    return {"path": str(configuration), "sha256": hashlib.sha256(raw).hexdigest()}


def require_cargo_audit_configuration(
    repo_root: Path, cargo_home: Path, expected: dict[str, str]
) -> None:
    """Refuse a Rust audit whose effective Cargo audit settings changed while it ran.

    Parameters
    ----------
    repo_root : Path
        Scanner working directory, whose .cargo/audit.toml takes precedence.
    cargo_home : Path
        Effective Cargo home, used only when project audit settings are absent.
    expected : dict of str
        Binding returned by :func:`cargo_audit_configuration` before the scanner ran.

    Raises
    ------
    OSError, ValueError
        The effective configuration appeared, disappeared, changed its bytes or
        became unqualified, so the report cannot be bound to reviewed settings.
    """
    if cargo_audit_configuration(repo_root, cargo_home) != expected:
        raise ValueError("Cargo audit settings changed during advisory execution.")
