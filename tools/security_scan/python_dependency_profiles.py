# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hashlocked Python audit inputs

"""Read every maintained Python lock without installing or filtering markers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REQUIRED_PROFILES = tuple(
    f"requirements/{name}.txt"
    for name in (
        "build",
        "ci-annealing",
        "ci-dev",
        "ci-mpi",
        "ci-optics",
        "ci-torch-cpu",
        "docs",
        "fuzz",
        "hdl",
        "hub",
        "lint",
        "maturin",
        "pre-commit",
        "release",
        "runtime",
        "sby",
        "security-scanners",
        "semgrep",
        "wheels-test",
        "workstation",
    )
)
CONSTRAINT_INPUT = "requirements/semgrep-overrides.txt"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?=\s|$)")


@dataclass(frozen=True)
class PinnedDependency:
    """Retain a locked identity, its marker, hashes and advisory-query version.

    The official Torch CPU profile queries upstream public-version advisories
    conservatively; the CPU build identity and hashes are never rewritten.
    Other local builds are refused. See PyPA's version-specifiers specification
    and https://docs.pytorch.org/get-started/previous-versions/.
    """

    name: str
    version: str
    audit_version: str
    marker: str | None
    hashes: tuple[str, ...]


@dataclass(frozen=True)
class DependencyProfile:
    """Bind all locked marker branches to their exact input-file digest."""

    path: str
    sha256: str
    dependencies: tuple[PinnedDependency, ...]


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
        else:
            lines.append(pending + line)
            pending = ""
    if pending:
        raise ValueError("Unterminated requirement continuation.")
    return lines


def read_dependency_profile(repo_root: Path, path: str) -> DependencyProfile:
    """Read a complete hashlocked profile, including inactive marker branches.

    Parameters
    ----------
    repo_root : Path
        Repository containing maintained requirement files.
    path : str
        Relative path of one profile.

    Returns
    -------
    DependencyProfile
        Immutable input digest and every pinned requirement.

    Raises
    ------
    ValueError
        A pin/hash is missing, syntax is unsupported, or a local build has no
        reviewed advisory mapping. No package installation or file mutation occurs.
    OSError
        The profile cannot be read.
    """
    raw = (repo_root / path).read_bytes()
    lines = _logical_lines(raw.decode("utf-8"))
    cpu_origin = f"--extra-index-url {CPU_INDEX}" in lines
    dependencies: list[PinnedDependency] = []
    for line in lines:
        if line == "--only-binary :all:":
            continue
        if line == f"--extra-index-url {CPU_INDEX}" and path == "requirements/ci-torch-cpu.txt":
            continue
        hashes = tuple(_HASH.findall(line))
        requirement = Requirement(_HASH.sub("", line).strip())
        specifiers = list(requirement.specifier)
        if requirement.url or len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError("Audit inputs must contain exact package versions.")
        version = Version(specifiers[0].version)
        if not hashes:
            raise ValueError("Every dependency must retain SHA-256 hashes.")
        name = canonicalize_name(requirement.name)
        audit_version = str(version)
        if version.local is not None:
            if not (
                path == "requirements/ci-torch-cpu.txt"
                and cpu_origin
                and name == "torch"
                and version.local == "cpu"
            ):
                raise ValueError("Local build has no reviewed advisory mapping.")
            audit_version = version.public
        dependencies.append(
            PinnedDependency(
                name,
                str(version),
                audit_version,
                str(requirement.marker) if requirement.marker is not None else None,
                hashes,
            )
        )
    if not dependencies:
        raise ValueError("An audit profile must contain pinned dependencies.")
    return DependencyProfile(path, hashlib.sha256(raw).hexdigest(), tuple(dependencies))


def discover_dependency_profiles(repo_root: Path) -> tuple[DependencyProfile, ...]:
    """Read required profiles and every additional requirements/*.txt lock.

    Parameters
    ----------
    repo_root : Path
        Repository to inventory, without Git or network access.

    Returns
    -------
    tuple of DependencyProfile
        Sorted profiles; only the reviewed Semgrep constraint file is excluded.

    Raises
    ------
    ValueError
        A maintained profile is absent or an override disagrees with its lock.
    OSError
        Any required input is unreadable. New profiles are audited automatically.
    """
    paths = {
        p.relative_to(repo_root).as_posix() for p in (repo_root / "requirements").glob("*.txt")
    }
    if not set(REQUIRED_PROFILES).issubset(paths) or CONSTRAINT_INPUT not in paths:
        raise ValueError("A maintained Python audit input is missing.")
    profiles = tuple(
        read_dependency_profile(repo_root, p) for p in sorted(paths - {CONSTRAINT_INPUT})
    )
    semgrep = next(p for p in profiles if p.path == "requirements/semgrep.txt")
    locked = {(d.name, d.version) for d in semgrep.dependencies}
    overrides = _logical_lines((repo_root / CONSTRAINT_INPUT).read_text(encoding="utf-8"))
    if not overrides:
        raise ValueError("Semgrep override input is empty.")
    for line in overrides:
        req = Requirement(line)
        specs = list(req.specifier)
        if req.url or req.marker or len(specs) != 1 or specs[0].operator != "==":
            raise ValueError("Semgrep overrides must be exact unconditional pins.")
        if (canonicalize_name(req.name), str(Version(specs[0].version))) not in locked:
            raise ValueError("Semgrep override is absent from the audited lock.")
    return profiles


def dependency_batches(profile: DependencyProfile) -> tuple[tuple[PinnedDependency, ...], ...]:
    """Partition all versions into executable batches without conflicting pins.

    Parameters
    ----------
    profile : DependencyProfile
        Complete input, retaining every marker branch.

    Returns
    -------
    tuple of tuple of PinnedDependency
        Batches with one query version per name. Exact query duplicates share a
        result; differing versions are audited separately. No host marker is applied.
    """
    unique = {(d.name, d.audit_version): d for d in profile.dependencies}
    batches: list[list[PinnedDependency]] = []
    for _, dep in sorted(unique.items()):
        for batch in batches:
            if all(d.name != dep.name for d in batch):
                batch.append(dep)
                break
        else:
            batches.append([dep])
    return tuple(tuple(batch) for batch in batches)
