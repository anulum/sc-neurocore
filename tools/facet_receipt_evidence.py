# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public receipt projection and exact private execution custody

"""Retain original execution artifacts and publish location-free projections."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

PROJECTION_CONTRACT = "repository-relative-execution.v1"
_RUNTIME_PREFIX = "facet_runtime."
_PUBLIC_RUNTIME = ("implementation", "python", "executable_sha256", "pytest")
_PRIVATE_RUNTIME = ("executable", "prefix", "hostname", "user", "home", "temporary")


class ExecutionStartError(RuntimeError):
    """An evidence command could not create its owned child process."""


@dataclass(frozen=True)
class CapturedExecution:
    """Retain exact output and independent command, interruption and cleanup facts.

    Attributes
    ----------
    exit_code : int
        Observed child exit status, or minus one when cleanup could not reap it.
    stdout : bytes
        Exact finite standard-output snapshot.
    stderr : bytes
        Exact finite standard-error snapshot.
    timed_out : bool
        The evidence command exceeded its execution budget.
    interruption : BaseException or None
        Original caller interruption; retained even if cleanup also times out.
    cleanup_error : subprocess.TimeoutExpired or None
        Independent failure to reap within the cleanup budget.
    """

    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    interruption: BaseException | None
    cleanup_error: subprocess.TimeoutExpired | None = None


def _captured_bytes(descriptor: int) -> bytes:
    """Read a fixed file extent without moving an inherited writer's file offset."""
    size = os.fstat(descriptor).st_size
    if size == 0:
        return b""
    if sys.platform == "win32":
        with mmap.mmap(descriptor, size, access=mmap.ACCESS_READ) as captured:
            return captured[:]
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(size - offset, 1024 * 1024), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


if sys.platform != "win32":

    def _wait_for_owned_group(
        process: subprocess.Popen[bytes], deadline: float, timeout: float
    ) -> None:
        """Wait within the original budget for the owned POSIX group after its leader exits."""
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            time.sleep(min(remaining, 1 / (1 + (os.cpu_count() or 1))))


def capture_execution(
    command: list[str], *, repo_root: Path, environment: Mapping[str, str], timeout: float
) -> CapturedExecution:
    """Capture a bounded command without depending on descendant pipe closure.

    Parameters
    ----------
    command : list[str]
        Exact child invocation, without a shell.
    repo_root : pathlib.Path
        Working directory of the evidence command.
    environment : Mapping[str, str]
        Explicit environment, including the actual-child provenance plugin.
    timeout : float
        Caller-admitted execution budget; cleanup receives the same finite budget.

    Returns
    -------
    CapturedExecution
        Exit status, exact output snapshots, any original interruption and
        any independent cleanup timeout.
        POSIX owns its newly created group; Windows owns the direct child.

    Raises
    ------
    ExecutionStartError
        If the child process could not start.
    OSError
        If private temporary output or an owned process cannot be accessed.
    """
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            raise ExecutionStartError("validator process could not be started") from error
        deadline = time.monotonic() + timeout
        exit_code = -1
        timed_out = False
        interruption: BaseException | None = None
        cleanup_error: subprocess.TimeoutExpired | None = None
        captured_stdout = b""
        captured_stderr = b""
        try:
            exit_code = process.wait(timeout=timeout)
            if sys.platform != "win32" and os.name == "posix":
                _wait_for_owned_group(process, deadline, timeout)
            captured_stdout = _captured_bytes(stdout.fileno())
            captured_stderr = _captured_bytes(stderr.fileno())
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException as error:
            interruption = error
        if timed_out or interruption is not None:
            if sys.platform != "win32" and os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                exit_code = -1
                cleanup_error = error
            captured_stdout = _captured_bytes(stdout.fileno())
            captured_stderr = _captured_bytes(stderr.fileno())
        return CapturedExecution(
            exit_code,
            captured_stdout,
            captured_stderr,
            timed_out,
            interruption,
            cleanup_error,
        )


class _JUnitElement(Protocol):
    """Mutable JUnit fields used after the defused parser accepts a report."""

    attrib: dict[str, str]
    tag: str
    text: str | None
    tail: str | None

    def iter(self, tag: str | None = None) -> Iterator[Self]:
        """Iterate descendants of the parsed element."""
        ...

    def get(self, key: str, default: str) -> str:
        """Read an attribute with an explicit textual default."""
        ...

    def set(self, key: str, value: str) -> None:
        """Replace a projected attribute."""
        ...

    def remove(self, child: Self) -> None:
        """Remove a private runtime property from its parent."""
        ...

    def __iter__(self) -> Iterator[Self]:
        """Iterate direct children of the parsed element."""
        ...


def private_archive_directory(repo_root: Path, requested: Path | None) -> Path:
    """Resolve raw custody outside public repository surfaces.

    Parameters
    ----------
    repo_root : Path
        Working repository whose public files must never contain raw identity.
    requested : Path or None
        Private archive override, or the ignored repository default.

    Returns
    -------
    pathlib.Path
        Archive parent in the private internal tree or outside the repository.

    Raises
    ------
    ValueError
        If the destination resolves inside a public repository directory.
    """
    repository = repo_root.resolve()
    directory = (requested or repository / "docs/internal/receipt_runs").resolve()
    if directory.is_relative_to(repository) and not directory.is_relative_to(
        repository / "docs/internal"
    ):
        raise ValueError("raw execution artifacts require a private archive directory")
    return directory


def logical_command(command: Sequence[str], repo_root: Path) -> tuple[str, ...]:
    """Describe the executed pytest invocation without an interpreter location.

    Parameters
    ----------
    command : Sequence[str]
        Actual direct pytest command, including its original selectors.
    repo_root : Path
        Working repository used to make absolute selectors relative.

    Returns
    -------
    tuple[str, ...]
        Logical interpreter or pytest name and equivalent repository selectors.

    Raises
    ------
    ValueError
        If an absolute argument is outside the working repository.
    """
    executable = Path(command[0]).name
    result = ["pytest" if executable in {"pytest", "pytest.exe"} else "python"]
    for argument in command[1:]:
        option, equals, value = argument.partition("=")
        prefix = option + equals if option.startswith("-") and equals else ""
        path, separator, selector = (value if prefix else argument).partition("::")
        if Path(path).is_absolute():
            try:
                path = Path(path).resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError as error:
                raise ValueError("validator argument is outside the repository") from error
            argument = prefix + path + (separator + selector if separator else "")
        result.append(argument)
    return tuple(result)


def _runtime_properties(root: _JUnitElement) -> dict[str, str]:
    """Read exactly one complete actual-child identity from its JUnit report."""
    properties: dict[str, str] = {}
    for item in root.iter("property"):
        name = item.get("name", "")
        if not name.startswith(_RUNTIME_PREFIX):
            continue
        key = name.removeprefix(_RUNTIME_PREFIX)
        if key in properties:
            raise ValueError("duplicate validator runtime property")
        properties[key] = item.get("value", "")
    if any(not properties.get(key) for key in (*_PUBLIC_RUNTIME, *_PRIVATE_RUNTIME)):
        raise ValueError("validator did not record complete runtime provenance")
    if re.fullmatch(r"[0-9a-f]{64}", properties["executable_sha256"]) is None:
        raise ValueError("invalid validator executable digest")
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", properties[key]) is None
        for key in _PUBLIC_RUNTIME
    ):
        raise ValueError("invalid public validator runtime fact")
    return properties


def _project_text(text: str, replacements: Sequence[tuple[str, str]]) -> str:
    """Replace known location metadata and refuse residual private locations."""
    for original, public in replacements:
        if public in {"<hostname>", "<user>"}:
            text = re.sub(r"(?<!\w)" + re.escape(original) + r"(?!\w)", public, text)
        else:
            text = text.replace(original, public)
    if re.search(
        r"(?:/home/|/media/|/tmp/|/var/tmp/|[A-Za-z]:[\\/]|\\\\[\w.-]+\\|(?<![\w:<>/])/(?:[\w.-]+/)*[\w.-]+)",
        text,
    ):
        raise ValueError("execution output contains an unclassified private location")
    return text


def _project_junit(root: _JUnitElement, replacements: Sequence[tuple[str, str]]) -> None:
    """Remove location metadata while preserving every testcase result and time."""
    for element in root.iter():
        element.attrib.pop("hostname", None)
        for key, value in list(element.attrib.items()):
            projected = _project_text(value, replacements)
            if projected != value:
                if element.tag == "testcase" and key in {"classname", "name"}:
                    raise ValueError("private identity occurs in a testcase identifier")
                if key in {"time", "timestamp", "tests", "failures", "errors", "skipped"}:
                    raise ValueError("private identity overlaps JUnit result accounting")
            element.set(key, projected)
        if element.text:
            element.text = _project_text(element.text, replacements)
        if element.tail:
            element.tail = _project_text(element.tail, replacements)
        for child in list(element):
            if (
                child.tag == "property"
                and child.get("name", "").removeprefix(_RUNTIME_PREFIX) in _PRIVATE_RUNTIME
                and child.get("name", "").startswith(_RUNTIME_PREFIX)
            ):
                element.remove(child)


def retain_execution(
    *,
    command: Sequence[str],
    executed_command: Sequence[str],
    repo_root: Path,
    temporary: Path,
    stdout: bytes,
    stderr: bytes,
    junit: bytes,
    exit_code: int,
    private_directory: Path,
) -> dict[str, str]:
    """Archive exact run bytes and return the documented public projection.

    Parameters
    ----------
    command : Sequence[str]
        Original user-selected command.
    executed_command : Sequence[str]
        Actual argv, including recorder-added plugin and JUnit arguments.
    repo_root : Path
        Actual execution working directory.
    temporary : Path
        Recorder-owned temporary JUnit directory.
    stdout, stderr, junit : bytes
        Exact captured artifacts, before decoding or normalisation.
    exit_code : int
        Observed validator process status, including a timeout sentinel.
    private_directory : Path
        Private raw-artifact parent, excluded from publication.

    Returns
    -------
    dict[str, str]
        Projected outputs, actual child runtime facts and exact private hashes.
        Projection failure returns a reason without exposing raw private text.
    """
    private_directory = private_archive_directory(repo_root, private_directory)
    private_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = Path(tempfile.mkdtemp(prefix="validator-", dir=private_directory))
    artifacts = {"stdout.bin": stdout, "stderr.bin": stderr, "junit.xml": junit}
    hashes = {}
    for name, payload in artifacts.items():
        target = archive / name
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "command": list(command),
        "executed_command": list(executed_command),
        "cwd": str(repo_root),
        "exit_code": exit_code,
        "artifacts": hashes,
        "projection": PROJECTION_CONTRACT,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with (archive / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "projection": PROJECTION_CONTRACT,
        "raw_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "raw_junit_sha256": hashes["junit.xml"],
        "raw_stdout_sha256": hashes["stdout.bin"],
        "raw_stderr_sha256": hashes["stderr.bin"],
    }
    try:
        root = ET.fromstring(junit, forbid_dtd=True, forbid_entities=True, forbid_external=True)
        runtime = _runtime_properties(root)
        result.update(
            version=runtime["pytest"],
            runtime_implementation=runtime["implementation"],
            runtime_python=runtime["python"],
            runtime_executable_sha256=runtime["executable_sha256"],
        )
        replacements = _replacements(runtime, repo_root, temporary)
        _project_junit(root, replacements)
        result.update(
            junit_xml=ET.tostring(root, encoding="unicode"),
            stdout=_project_text(stdout.decode("utf-8", errors="replace"), replacements),
            stderr=_project_text(stderr.decode("utf-8", errors="replace"), replacements),
        )
    except (ET.ParseError, DefusedXmlException, ValueError) as error:
        result["provenance_error"] = (
            "invalid private JUnit report"
            if isinstance(error, (ET.ParseError, DefusedXmlException))
            else str(error)
        )
    return result


def _replacements(
    runtime: Mapping[str, str], repo_root: Path, temporary: Path
) -> tuple[tuple[str, str], ...]:
    """Map actual child and recorder location metadata to descriptive tokens."""
    locations = {
        str(repo_root): "<repository>",
        str(repo_root.resolve()): "<repository>",
        str(temporary): "<temporary>",
        runtime["temporary"]: "<temporary-root>",
        runtime["executable"]: "<python-executable>",
        runtime["prefix"]: "<python-environment>",
        runtime["home"]: "<home>",
    }
    replacements = sorted(locations.items(), key=lambda item: len(item[0]), reverse=True)
    replacements.extend((runtime[key], "<" + key + ">") for key in ("hostname", "user"))
    return tuple(replacements)
