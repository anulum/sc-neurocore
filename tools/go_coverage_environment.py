# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact inherited original-ABI child environment declarations

"""Conserve real Pytest metadata without ignoring any inherited environment field.

The original crash-fenced calls inherit their parent's environment. The parent
declares that complete mapping in its flushed audit record before spawning;
the child resolves it by its actual PPID and argv before native loading.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import os
import sys
import traceback
from typing import cast

_PYTEST_FIELDS = frozenset({"PYTEST_VERSION", "PYTEST_CURRENT_TEST"})


def terminal_environment() -> dict[str, str]:
    """Declare the real Readline terminal metadata before any consumer starts.

    Returns
    -------
    dict
        Actual positive ASCII COLUMNS and LINES values from the C environment
        after initializing the installed Readline module used by Pytest capture.

    Raises
    ------
    ModuleNotFoundError
        The executing Linux interpreter does not provide Readline.
    RuntimeError
        The real C environment omits a terminal dimension or contains a value
        that is not a canonical positive ASCII integer.

    Notes
    -----
    Readline may update libc's environment without updating Python's cached
    ``os.environ``. Declare its actual values explicitly; never invent a
    terminal size, inherit unrelated fields or remove keys during validation.
    """
    importlib.import_module("readline")
    getter = ctypes.CDLL(None).getenv
    getter.argtypes = [ctypes.c_char_p]
    getter.restype = ctypes.c_char_p
    result: dict[str, str] = {}
    for key in ("COLUMNS", "LINES"):
        raw = getter(key.encode("ascii"))
        if not isinstance(raw, bytes) or re.fullmatch(rb"[1-9][0-9]*", raw) is None:
            raise RuntimeError("Actual Readline terminal field is absent or invalid: " + key)
        result[key] = raw.decode("ascii")
    return result


def declare_terminal_environment(job: Path) -> dict[str, str]:
    """Retain the real first Readline initialization and refuse later metadata drift.

    Parameters
    ----------
    job : pathlib.Path
        Actual owned gate allocation receiving the exclusive initialization record.

    Returns
    -------
    dict
        The same actual C terminal pair declared for every consumer in this job.

    Raises
    ------
    RuntimeError
        Actual terminal values or the imported Readline image changed after the
        first declaration. Invalid native values are refused by the reader too.
    """
    loaded_before = "readline" in sys.modules
    terminal = terminal_environment()
    module_path = Path(str(sys.modules["readline"].__file__)).resolve(strict=True)
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    path = job / "readline-initialization.json"
    if path.exists():
        original = json.loads(path.read_text())
        if (
            original.get("terminal_environment") != terminal
            or original.get("module_path") != str(module_path)
            or original.get("module_sha256") != module_sha
        ):
            raise RuntimeError("Actual Readline initialization changed after declaration")
    else:
        record = {
            "producer_pid": os.getpid(),
            "module_loaded_before": loaded_before,
            "module_path": str(module_path),
            "module_sha256": module_sha,
            "terminal_environment": terminal,
            "python_cache_matches": {
                key: os.environ.get(key) == value for key, value in terminal.items()
            },
            "actual_declaration_stack": [
                {"file": frame.filename, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_stack()
            ],
        }
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    return terminal


def environment_mapping(value: object) -> dict[str, str]:
    """Require a complete mapping of actual environment names and string values.

    Parameters
    ----------
    value : object
        Decoded declaration or actual process environment.

    Returns
    -------
    dict
        The complete string mapping, with no removed keys.

    Raises
    ------
    ValueError
        The declaration is not a mapping of strings to strings.
    """
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("The native consumer environment must map strings to strings")
    return cast(dict[str, str], value)


def environment_digest(environment: dict[str, str]) -> str:
    """Hash the entire declared mapping using a stable representation, without filtering.

    Parameters
    ----------
    environment : dict
        Complete declared environment names and values.

    Returns
    -------
    str
        SHA256 of the canonical full mapping.
    """
    data = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def validate_child_environment(job: Path, base: dict[str, str], value: object) -> dict[str, str]:
    """Bind only the two real Pytest fields to the actual full original collection.

    Parameters
    ----------
    job : pathlib.Path
        Actual allocation retaining the original complete collection.
    base : dict
        Exact reviewed base environment supplied before Pytest starts.
    value : object
        Complete mapping captured by the real parent before spawning.

    Returns
    -------
    dict
        Conserved base and the two exact framework metadata fields.

    Raises
    ------
    ValueError
        A base value, key, version, collection or active call differs.
    """
    actual = environment_mapping(value)
    if set(actual) != set(base) | _PYTEST_FIELDS or any(
        actual.get(key) != item for key, item in base.items()
    ):
        raise ValueError("Original ABI child environment differs from its exact declared base")
    if actual["PYTEST_VERSION"] != importlib.metadata.version("pytest"):
        raise ValueError("Original ABI child Pytest version differs from the actual installed one")
    raw: object = json.loads((job / "collection.json").read_text())
    if (
        not isinstance(raw, list)
        or len(raw) != 630
        or any(not isinstance(node, str) for node in raw)
        or len(set(raw)) != 630
    ):
        raise ValueError("Original ABI child requires the complete real 630-case collection")
    marker = actual["PYTEST_CURRENT_TEST"]
    if (
        not marker.endswith(" (call)")
        or marker.removesuffix(" (call)") not in raw
        or "::test_null_pointer_refusal_is_crash_fenced[" not in marker
    ):
        raise ValueError("Original ABI child marker is not an actual crash-fenced call")
    return actual


def child_declaration(job: Path, base: dict[str, str], actual: dict[str, str]) -> dict[str, object]:
    """Declare the actual inherited mapping before the real original child is spawned.

    Parameters
    ----------
    job : pathlib.Path
        Actual gate allocation supplying the conserved collection.
    base : dict
        Reviewed base mapping of the real test parent.
    actual : dict
        Full inherited mapping observed at the real process audit event.

    Returns
    -------
    dict
        The complete mapping and digest to flush before the original spawn.
    """
    environment = validate_child_environment(job, base, actual)
    return {
        "child_environment": environment,
        "child_environment_sha256": environment_digest(environment),
    }


def read_child_environment(
    job: Path, base: dict[str, str], parent_pid: int, argv: list[str]
) -> dict[str, str]:
    """Resolve one flushed real-parent declaration by actual PPID and exact child argv.

    Parameters
    ----------
    job : pathlib.Path
        Actual allocation retaining the parent's append-only audit stream.
    base : dict
        Original reviewed base environment, without framework metadata.
    parent_pid : int
        The running child's actual PPID, or an observed PID during reconciliation.
    argv : list of str
        The child's exact observed Python arguments.

    Returns
    -------
    dict
        One source-bound full startup declaration, without ignored fields.

    Raises
    ------
    ValueError
        Parent identity, launch uniqueness, environment or digest differs.
    """
    path = job / "process-counters/tests" / str(parent_pid) / "observations.jsonl"
    raw: list[object] = [json.loads(line) for line in path.read_text().splitlines()]
    if not raw or any(not isinstance(row, dict) for row in raw):
        raise ValueError("Original ABI child parent stream is empty or malformed")
    rows = cast(list[dict[str, object]], raw)
    if rows[0].get("event") != "registered" or rows[0].get("pid") != parent_pid:
        raise ValueError("Original ABI child parent registration differs from its actual PPID")
    matching = [
        row
        for row in rows
        if row.get("event") == "abi_child_launch"
        and row.get("parent_pid") == parent_pid
        and row.get("expected_child_argv") == argv
    ]
    if len(matching) != 1:
        raise ValueError("Original ABI child has no unique exact parent launch declaration")
    declaration = matching[0]
    environment = validate_child_environment(job, base, declaration.get("child_environment"))
    if declaration.get("child_environment_sha256") != environment_digest(environment):
        raise ValueError("Original ABI child environment declaration digest differs")
    return environment
