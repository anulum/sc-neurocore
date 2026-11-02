# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External formal receipt consistency checks

"""Check external proof evidence without claiming to reproduce its execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class EvidencePolicy:
    """Identify the expected machine and its admitted measurement limits.

    Parameters
    ----------
    hostname : str
        Actual machine hostname, not a display label supplied by the receipt.
    memory_bytes : int
        Exact admitted cgroup memory limit.
    timeout_seconds : int
        Exact admitted primary-service runtime ceiling.
    label : str
        Display name; focused tests identify their local qualification explicitly.
    audit_memory_bytes : int
        Separate assertion-enable audit's admitted cgroup memory limit.
    audit_timeout_seconds : int
        Separate assertion-enable audit's admitted runtime ceiling.
    """

    hostname: str = "god-of-the-math"
    memory_bytes: int = 64 * 1024**3
    timeout_seconds: int = 7200
    label: str = "ML350"
    audit_memory_bytes: int = 8 * 1024**3
    audit_timeout_seconds: int = 1800


def object_fields(value: object) -> dict[str, object]:
    """Require a JSON object with string keys.

    Parameters
    ----------
    value : object
        Decoded JSON value.

    Returns
    -------
    dict of str to object
        Checked object fields.

    Raises
    ------
    ValueError
        If the value is not an object with string keys.
    """
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected an object with string keys.")
    return {key: item for key, item in value.items()}


def read_object(path: Path) -> dict[str, object]:
    """Read a JSON object from a real file.

    Parameters
    ----------
    path : Path
        Input file.

    Returns
    -------
    dict of str to object
        Decoded object.
    """
    return object_fields(json.loads(path.read_text(encoding="utf-8")))


def confined_file(repository: Path, value: object) -> Path:
    """Resolve an existing evidence file within its checkout.

    Parameters
    ----------
    repository : Path
        Checkout root.
    value : object
        Repository-relative path.

    Returns
    -------
    Path
        Resolved regular file.

    Raises
    ------
    ValueError
        If a path is absolute, escapes the checkout or is not a file.
    """
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("Evidence paths must be repository-relative strings.")
    path = (repository / value).resolve(strict=True)
    if not path.is_relative_to(repository) or not path.is_file():
        raise ValueError("Evidence must be a regular file inside the checkout.")
    return path


def digest(path: Path) -> str:
    """Compute the digest of actual file bytes.

    Parameters
    ----------
    path : Path
        Input file.

    Returns
    -------
    str
        Lowercase SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, reason: str) -> None:
    """Reject a failed receipt condition.

    Parameters
    ----------
    condition : bool
        Condition that must hold.
    reason : str
        Concrete refusal reason.

    Raises
    ------
    ValueError
        If the condition is false.
    """
    if not condition:
        raise ValueError(reason)


def finite_number(value: object, name: str) -> float:
    """Require a finite numeric measurement, excluding booleans.

    Parameters
    ----------
    value : object
        Measurement value.
    name : str
        Field name for a refusal.

    Returns
    -------
    float
        Checked finite measurement.

    Raises
    ------
    ValueError
        If the value is nonnumeric, boolean or nonfinite.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite.")
    return result


def source_pins(provenance: str) -> dict[str, str]:
    """Read immutable source pins from the current SDK provenance.

    Parameters
    ----------
    provenance : str
        Verified SDK provenance text.

    Returns
    -------
    dict of str to str
        Yosys and SymbiYosys source commits.

    Raises
    ------
    ValueError
        If either source identity is missing or ambiguous.
    """
    result = {}
    for name, pattern in (
        ("yosys_source", r"^Yosys [^\n]*git sha1 ([0-9a-f]{40}),"),
        ("sby_source", r"^SBY source ([0-9a-f]{40})$"),
    ):
        matches = re.findall(pattern, provenance, flags=re.MULTILINE)
        require(len(matches) == 1, f"SDK {name} must have one immutable identity.")
        result[name] = matches[0]
    return result


def validate_measurement(
    receipt: dict[str, object], policy: EvidencePolicy, wall_seconds: float
) -> tuple[float, int]:
    """Check terminal success, accounting and process-tree closure.

    Parameters
    ----------
    receipt : dict of str to object
        Actual native receipt.
    policy : EvidencePolicy
        Required machine and resource contract.
    wall_seconds : float
        Wall already checked against its actual native clock stamps.

    Returns
    -------
    tuple of float and int
        Actual wall seconds and true cgroup peak bytes.

    Raises
    ------
    ValueError
        If execution, accounting, resource limits or closure do not qualify.
    """
    measurement = object_fields(receipt.get("measurement"))
    for field, expected in {
        "hostname": policy.hostname,
        "status": "PASS",
        "active_state": "inactive",
        "cgroup_processes": [],
    }.items():
        require(measurement.get(field) == expected, f"Native {field} does not qualify.")
    require(measurement.get("process_tree_closed") is True, "Native process tree is not closed.")
    for field, expected_limit in {
        "main_pid": 0,
        "memory_limit_bytes": policy.memory_bytes,
        "swap_limit_bytes": 0,
        "time_limit_seconds": policy.timeout_seconds,
    }.items():
        value = measurement.get(field)
        require(type(value) is int and value == expected_limit, f"Native {field} does not qualify.")
    invocation = measurement.get("invocation_id")
    require(
        isinstance(invocation, str) and re.fullmatch(r"[0-9a-f]{32}", invocation) is not None,
        "Native invocation identity is missing or invalid.",
    )
    events = object_fields(measurement.get("oom_events"))
    for field in ("oom", "oom_kill", "oom_group_kill"):
        value = events.get(field)
        require(type(value) is int and value == 0, f"Native {field} must be zero.")
    wall = finite_number(wall_seconds, "wall_seconds")
    peak = finite_number(measurement.get("peak_memory_bytes"), "peak_memory_bytes")
    require(0 < wall <= policy.timeout_seconds, "Native wall is outside the admitted limit.")
    require(
        type(measurement.get("peak_memory_bytes")) is int and 0 < peak <= policy.memory_bytes,
        "Native cgroup peak is outside the admitted limit.",
    )
    return wall, int(peak)
