# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native checkpoint export timeout custody

"""Verify original native execution and preserve a timed-out export before recovery."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

from tools.function_synthesis_toolchain import SynthesisToolchain

_TIMEOUT_DIRECTORY = "initial-export-timeout"
_ORIGINAL_INPUTS = {"unit.v", "physical-module.json"}


def validate_checkpoint_command(
    directory: Path, stem: str, tools: SynthesisToolchain, command: list[str], cap: float
) -> None:
    """Validate the complete original argument vector and the engine's own deadline."""
    value: object = json.loads((directory / (stem + ".command.json")).read_text())
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("A native checkpoint command requires its original argument vector")
    argv = [item for item in value if isinstance(item, str)]
    if (
        len(argv) != 7 + len(command)
        or argv[:6]
        != [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
        ]
        or argv[7:] != command
    ):
        raise ValueError("The retained checkpoint command differs from the bounded native export")
    seconds = float(argv[6])
    if not math.isfinite(seconds) or not 0 < seconds <= cap:
        raise ValueError("The retained checkpoint command exceeds its native deadline")


def _native_history(local: Path) -> tuple[Path, ...]:
    """Bind only the optional regular Yosys history created below the native home."""
    current = local
    for component in ("state", "yosys", "history"):
        if current.is_symlink() or not current.is_dir():
            raise ValueError("The initial checkpoint contains an invalid native history directory")
        expected = current / component
        children = tuple(current.iterdir())
        if not children:
            return ()
        if children != (expected,):
            raise ValueError("The initial checkpoint contains unexpected native history entries")
        current = expected
    if current.is_symlink() or not current.is_file():
        raise ValueError("The initial checkpoint contains nonregular native history")
    return (current,)


def _initial_files(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Verify original command artifacts and the one recognised native scratch hierarchy."""
    files: list[Path] = []
    for path in paths:
        if path.name == ".local":
            files.extend(_native_history(path))
        elif path.is_symlink() or not path.is_file():
            raise ValueError("The initial native checkpoint attempt contains a nonregular artifact")
        else:
            files.append(path)
    return tuple(files)


def preserve_checkpoint_timeout(directory: Path) -> None:
    """Move the original attempt while keeping its exact source files for the staged export."""
    original = tuple(
        sorted(path for path in directory.iterdir() if path.name not in _ORIGINAL_INPUTS)
    )
    _initial_files(original)
    attempt = directory / _TIMEOUT_DIRECTORY
    attempt.mkdir(exist_ok=False)
    for path in original:
        path.rename(attempt / path.name)


def read_checkpoint_timeout(
    directory: Path, flows: Mapping[str, str], tools: SynthesisToolchain
) -> tuple[str, tuple[Path, ...]] | None:
    """Bind an exact leading prefix of the initial legacy plan whose last command timed out."""
    attempt = directory / _TIMEOUT_DIRECTORY
    if not attempt.exists() and not attempt.is_symlink():
        return None
    if attempt.is_symlink() or not attempt.is_dir():
        raise ValueError("The original native checkpoint timeout requires a regular directory")
    paths = _initial_files(tuple(sorted(attempt.iterdir())))
    files = {path.name for path in paths if path.parent == attempt}
    if _ORIGINAL_INPUTS & files:
        raise ValueError("A native checkpoint timeout must retain the shared original inputs")
    for stem, expected in flows.items():
        script = stem + ".ys"
        if script not in files or (attempt / script).read_text(encoding="utf-8") != expected:
            raise ValueError("The original timed-out checkpoint script changed")
    invoked = [stem for stem in flows if {stem + ".command.json", stem + ".log"} & files]
    receipts = {stem + suffix for stem in invoked for suffix in (".log", ".command.json")}
    logs = {name for name in files if name.endswith((".log", ".command.json"))}
    if not invoked or invoked != list(flows)[: len(invoked)] or logs != receipts:
        raise ValueError("The original checkpoint did not retain an exact executed plan prefix")
    diagnostic = re.compile(
        rf"^(?:{re.escape(str(tools.deadline))}|{re.escape(tools.deadline.name)}): "
        r"sending signal KILL to command ",
        re.M,
    )
    for stem in invoked:
        validate_checkpoint_command(
            attempt, stem, tools, [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"], 90
        )
        expired = diagnostic.search((attempt / (stem + ".log")).read_text(encoding="utf-8"))
        if bool(expired) != (stem == invoked[-1]):
            raise ValueError("The initial checkpoint did not end at its verified native deadline")
    return invoked[-1], paths
