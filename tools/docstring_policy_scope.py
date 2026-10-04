# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — docstring policy cohort and threshold comparison

"""Validate docstring policy structure and preserve its trusted source cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
import sys
from typing import cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class PolicyError(ValueError):
    """Report an unavailable or malformed policy input using authored text."""


@dataclass(frozen=True)
class FilePolicy:
    """Record module documentation and permitted missing symbols for one file."""

    require_module: bool
    allow_missing: frozenset[str]


@dataclass(frozen=True)
class DocstringPolicy:
    """Bind the parsed file rules and minimum length to the input bytes."""

    minimum_chars: int
    files: dict[str, FilePolicy]
    sha256: str


def _table(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PolicyError("Policy tables must use string keys.")
    return cast(dict[str, object], value)


def _positive_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError("Policy counts and minimum lengths must be positive integers.")
    return value


def parse_policy(raw: bytes) -> DocstringPolicy:
    """Parse a nonempty policy with unique repository-relative Python paths.

    Parameters
    ----------
    raw : bytes
        Exact UTF-8 TOML policy bytes.

    Returns
    -------
    DocstringPolicy
        Validated rules and the SHA-256 of the supplied policy.

    Raises
    ------
    PolicyError
        If the input is malformed or its count disagrees with its unique cohort.
    """
    try:
        data = _table(tomllib.loads(raw.decode("utf-8")))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PolicyError("Docstring policy must be valid UTF-8 TOML.") from exc
    quality = _table(data.get("quality"))
    minimum = _positive_integer(quality.get("min_docstring_chars"))
    count = _positive_integer(quality.get("expected_file_count"))
    entries = data.get("file")
    if not isinstance(entries, list) or len(entries) != count:
        raise PolicyError("Docstring policy count must match its nonempty file cohort.")
    files: dict[str, FilePolicy] = {}
    for value in entries:
        entry = _table(value)
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path
            or "\\" in path
            or any(ord(char) < 32 for char in path)
            or not path.endswith(".py")
            or path in files
        ):
            raise PolicyError("Policy paths must be unique relative Python file paths.")
        module = entry.get("require_module_docstring", True)
        missing = entry.get("allow_missing", [])
        if not isinstance(module, bool) or not isinstance(missing, list):
            raise PolicyError("Module requirements and missing-symbol lists have invalid types.")
        if any(not isinstance(name, str) or not name.strip() for name in missing):
            raise PolicyError("Missing-symbol allowances must contain nonempty names.")
        names = cast(list[str], missing)
        if len(names) != len(set(names)):
            raise PolicyError("Missing-symbol allowances must not contain duplicate names.")
        files[path] = FilePolicy(module, frozenset(names))
    return DocstringPolicy(minimum, files, hashlib.sha256(raw).hexdigest())


def compare_policies(
    candidate: DocstringPolicy, baseline: DocstringPolicy, changed_python: set[str]
) -> list[str]:
    """Reject cohort omissions, weaker documentation rules and new allowances.

    Parameters
    ----------
    candidate : DocstringPolicy
        Current working-tree policy.
    baseline : DocstringPolicy
        Policy read from the trusted original Git commit.
    changed_python : set of str
        All added or modified Python paths, including untracked maintained files.

    Returns
    -------
    list of str
        Deterministic authored diagnostics; an empty list means scope acceptance.
    """
    violations: list[str] = []
    if candidate.minimum_chars < max(20, baseline.minimum_chars):
        violations.append("Docstring minimum length must not decrease below the trusted floor.")
    for path in sorted(set(baseline.files) | changed_python):
        if path not in candidate.files:
            violations.append(f"Missing required policy file: {path}")
    for path, rule in sorted(candidate.files.items()):
        prior = baseline.files.get(path, FilePolicy(True, frozenset()))
        if prior.require_module and not rule.require_module:
            violations.append(f"Module documentation requirement weakened: {path}")
        if not rule.allow_missing <= prior.allow_missing:
            violations.append(f"New missing-symbol allowance: {path}")
    return violations
