# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded synthesis tool execution and source custody

"""Inspect the actual synthesis executables and libraries used by a measurement."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


class SynthesisToolError(RuntimeError):
    """Report a native command which did not complete successfully.

    Parameters
    ----------
    returncode : int
        Actual native exit status.
    deadline_elapsed : bool, optional
        Whether the retained GNU timeout diagnostic explicitly reports its
        deadline signal. A status of 124 or 137 alone is not this evidence.

    """

    def __init__(self, returncode: int, deadline_elapsed: bool = False) -> None:
        """Retain the native status without exposing private command paths."""
        self.returncode = returncode
        self.deadline_elapsed = deadline_elapsed
        super().__init__(f"Synthesis tool command exited with status {returncode}")

    @property
    def timed_out(self) -> bool:
        """Return whether GNU timeout explicitly reported its elapsed deadline."""
        return self.deadline_elapsed


def _executable(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"Required synthesis executable unavailable: {name}")
    return Path(found).resolve()


def _environment(command: Path, directory: Path) -> dict[str, str]:
    """Exclude ambient code injection while retaining the caller's confined temporary root.

    Loader injection (``LD_PRELOAD``, ``LD_AUDIT`` and any other ``LD_*`` setting)
    is refused. ``LD_LIBRARY_PATH`` is a library search path that standard Python
    installations set for their own interpreter; it is accepted for the caller but,
    like every other ambient setting, never reaches the native child.
    """
    if any(name.startswith("LD_") and name != "LD_LIBRARY_PATH" for name in os.environ):
        raise ValueError("Native synthesis refuses dynamic-loader environment overrides")
    return {
        "PATH": str(command.parent) + os.pathsep + os.defpath,
        "HOME": str(directory),
        "TMPDIR": os.environ.get("TMPDIR", str(directory)),
        "LC_ALL": "C",
        "COLUMNS": "32768",
    }


def _stop_group(process: subprocess.Popen[bytes]) -> int:
    """Terminate the owned session's process group and reap its direct child within grace."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1.0)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    return process.wait(timeout=1.0)


def _execute(
    command: list[str],
    directory: Path,
    log: Path,
    seconds: float,
    deadline: Path,
    limits: Path,
) -> None:
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("A synthesis command requires a finite positive timeout")
    environment = _environment(Path(command[0]), directory)
    argv = [
        str(limits),
        "--core=0",
        "--",
        str(deadline),
        "--signal=KILL",
        "--verbose",
        str(seconds),
        *command,
    ]
    with log.with_suffix(".command.json").open("x", encoding="utf-8") as receipt:
        receipt.write(json.dumps(argv, indent=2) + "\n")
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            argv,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            # Yosys reads ABC's full readline command echo to detect completion.
            env=environment,
        )
        try:
            returncode = process.wait(timeout=seconds + 2.0)
        except subprocess.TimeoutExpired as error:
            returncode = _stop_group(process)
            raise SynthesisToolError(returncode) from error
        except BaseException:
            _stop_group(process)
            raise
    if returncode:
        elapsed = (
            returncode in {-9, 124, 137}
            and re.search(
                rf"^(?:{re.escape(str(deadline))}|{re.escape(deadline.name)}): "
                r"sending signal KILL to command ",
                log.read_text(encoding="utf-8"),
                re.M,
            )
            is not None
        )
        raise SynthesisToolError(returncode, elapsed)


def _native_yosys(
    requested: Path, directory: Path, seconds: float, deadline: Path, limits: Path
) -> Path:
    log = directory / "yosys-executable.log"
    _execute(
        [str(requested), "-Q", "-T", "-p", "exec -- /usr/bin/readlink /proc/$PPID/exe"],
        directory,
        log,
        seconds,
        deadline,
        limits,
    )
    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if Path(line).is_absolute()
    ]
    if len(lines) != 1:
        raise ValueError("Yosys output must identify exactly one absolute native executable")
    return _executable(lines[0])


def _data_root(log: Path) -> Path:
    paths = re.findall(
        r"Parsing (?:System)?Verilog input from .(.+/xilinx/cells_sim[.]v)' to AST representation[.]",
        log.read_text(encoding="utf-8"),
    )
    roots = {Path(path).resolve().parent.parent for path in paths}
    if len(roots) != 1:
        raise ValueError("Yosys output must identify exactly one technology-library root")
    return roots.pop()


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprints(
    executables: tuple[tuple[str, Path], ...], data: Path
) -> tuple[tuple[str, str], ...]:
    paths = [(f"bin/{name}", path) for name, path in executables]
    paths.extend(
        (f"data/{path.relative_to(data).as_posix()}", path)
        for path in sorted(data.rglob("*"))
        if path.is_file()
    )
    result = []
    for name, path in paths:
        result.append((name, _digest(path)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SynthesisToolchain:
    """Bind real executables, observed versions and the installed Yosys data tree.

    Parameters
    ----------
    yosys, abc, deadline, limits : pathlib.Path
        Resolved Yosys, its sibling yosys-abc, GNU timeout and prlimit binaries.
        The same ABC binary is used by mapping and explicit equivalence checks.
    data_directory : pathlib.Path
        Library root identified by an actual Yosys library-read command.
    fingerprints : tuple of (str, str)
        SHA-256 of all four executables and every installed data file.
    versions : tuple of (str, str)
        Version strings observed from the inspected executables.
    cohort_deadline : float or None, optional
        Absolute monotonic deadline shared by commands in one proof attempt.
        Runtime admission context; excluded from the installed tool identity.

    """

    yosys: Path
    abc: Path
    deadline: Path
    limits: Path
    data_directory: Path
    fingerprints: tuple[tuple[str, str], ...]
    versions: tuple[tuple[str, str], ...]
    cohort_deadline: float | None = None

    @property
    def executables(self) -> tuple[tuple[str, Path], ...]:
        """Return the exact binaries used by the measurement and its containment."""
        return (
            ("yosys", self.yosys),
            ("abc", self.abc),
            ("timeout", self.deadline),
            ("prlimit", self.limits),
        )

    @classmethod
    def inspect(
        cls,
        directory: Path,
        *,
        yosys: str = "yosys",
        deadline: str = "timeout",
        limits: str = "prlimit",
        timeout: float = 30.0,
    ) -> SynthesisToolchain:
        """Inspect an existing toolchain without installing or replacing tools.

        Parameters
        ----------
        directory : pathlib.Path
            New private directory retaining the commands and actual output.
        yosys, deadline, limits : str, optional
            Existing executable names or paths. An actual Yosys child identifies
            its running parent through Linux procfs, including behind launchers.
            ABC is taken from that native binary's directory, matching mapping.
        timeout : float, optional
            Finite positive deadline for each inspection command.

        Returns
        -------
        SynthesisToolchain
            Executable, version and complete technology-library bindings.

        Raises
        ------
        OSError
            For missing tools, libraries or an already existing receipt directory.
        ValueError
            For invalid deadlines, ambiguous executable or library output,
            or empty versions.
        SynthesisToolError
            If an actual inspection command fails or reaches its deadline.

        """
        requested = _executable(yosys)
        deadline_path, limits_path = _executable(deadline), _executable(limits)
        directory.mkdir(parents=True, exist_ok=False)
        yosys_path = _native_yosys(requested, directory, timeout, deadline_path, limits_path)
        abc_path = _executable(str(yosys_path.parent / "yosys-abc"))
        binaries = (
            ("yosys", yosys_path),
            ("abc", abc_path),
            ("timeout", deadline_path),
            ("prlimit", limits_path),
        )
        executable_pins = tuple((f"bin/{name}", _digest(path)) for name, path in binaries)
        observed: list[tuple[str, str]] = []
        for name, executable in binaries:
            args = ["-s", "-c", "version"] if name == "abc" else ["--version"]
            log = directory / f"{name}-version.log"
            _execute(
                [str(executable), *args],
                directory,
                log,
                timeout,
                deadline_path,
                limits_path,
            )
            lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
            if not lines:
                raise ValueError(f"Empty version output from {name}")
            observed.append((name, lines[-1] if name == "abc" else lines[0]))
        library_log = directory / "yosys-library.log"
        _execute(
            [str(yosys_path), "-Q", "-T", "-p", "read_verilog -lib +/xilinx/cells_sim.v"],
            directory,
            library_log,
            timeout,
            deadline_path,
            limits_path,
        )
        data = _data_root(library_log)
        toolchain = cls(
            yosys_path,
            abc_path,
            deadline_path,
            limits_path,
            data,
            executable_pins + _fingerprints(binaries, data)[len(binaries) :],
            tuple(observed),
        )
        toolchain.verify()
        return toolchain

    def manifest(self) -> dict[str, object]:
        """Return path-independent executable and library provenance.

        Returns
        -------
        dict of str to object
            Actual version strings and SHA-256 bindings. Private host paths
            are retained only in the raw command receipts.

        """
        return {"versions": dict(self.versions), "sha256": dict(self.fingerprints)}

    @property
    def identity(self) -> str:
        """Return the SHA-256 of the canonical toolchain provenance manifest."""
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def verify(self) -> None:
        """Refuse changed executables or an altered installed technology-data tree.

        Raises
        ------
        OSError
            If a bound file is no longer available.
        ValueError
            If any executable, library byte or data-file inventory has changed.

        """
        if _fingerprints(self.executables, self.data_directory) != self.fingerprints:
            raise ValueError("Synthesis toolchain changed after inspection")

    def check_library_log(self, log: Path) -> None:
        """Require an actual mapping log to use the inspected technology library.

        Parameters
        ----------
        log : pathlib.Path
            Retained Yosys output from the mapping command.

        Raises
        ------
        OSError, ValueError
            If the output cannot identify the exact inspected library root.

        """
        if _data_root(log) != self.data_directory:
            raise ValueError("Mapping used a different Yosys technology library")

    def run(
        self,
        command: list[str],
        directory: Path,
        log: Path,
        *,
        timeout: float,
    ) -> None:
        """Run an inspected Yosys or ABC command with retained output and a deadline.

        Parameters
        ----------
        command : list of str
            Argument vector beginning with the exact inspected Yosys or ABC.
        directory : pathlib.Path
            Existing private working directory.
        log : pathlib.Path
            Retained combined stdout/stderr; a neighbouring command JSON retains
            the actual containment and tool argument vector.
        timeout : float
            Finite positive deadline. GNU timeout terminates the command's process
            group with SIGKILL at the deadline; prlimit disables child cores.
            The caller supplies the CPU and memory containment envelope.

        Raises
        ------
        OSError, ValueError
            If paths, deadlines, executable identity or file bindings are invalid.
        SynthesisToolError
            If the tool fails or its elapsed deadline is reached.

        """
        if not command or Path(command[0]).resolve() not in (self.yosys, self.abc):
            raise ValueError("Only inspected Yosys or ABC executables may run")
        executable = Path(command[0]).resolve()
        args = ["-s", *command[1:]] if executable == self.abc else command[1:]
        self.verify()
        if self.cohort_deadline is not None and (
            not math.isfinite(self.cohort_deadline)
            or time.monotonic() + timeout > self.cohort_deadline
        ):
            raise ValueError("Insufficient shared proof cohort budget for the native command")
        _execute([str(executable), *args], directory, log, timeout, self.deadline, self.limits)
        self.verify()
        if self.cohort_deadline is not None and time.monotonic() > self.cohort_deadline:
            raise ValueError("Native command validation exhausted the shared proof cohort budget")
