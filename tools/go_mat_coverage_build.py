# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native Go coverage producers and command custody

"""Build and inspect real MAT images inside an already admitted caller allocation.

This module does not admit resources, change cooling or create a supervisor.
The caller supplies its complete deadline, compile concurrency and environment.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from tools.go_coverage_build_inputs import (
    validate_gofmt_output,
    validate_native_dependencies,
    validate_native_exports,
    validate_native_test_events,
    validate_native_test_names,
)
from tools.go_coverage_profiles import MODULE
from tools.go_coverage_trace import validate_config


def save(path: Path, value: object) -> None:
    """Create a new immutable artifact, refusing stale or colliding output paths."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def digest(path: Path) -> str:
    """Hash the complete real backing file without substituting path metadata."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class NativeGo:
    """Retain each real command and native artifact in separate exclusive namespaces."""

    def __init__(
        self,
        job: Path,
        repository: Path,
        go: Path,
        deadline: float,
        jobs: int,
        environment: dict[str, str],
        stop_seconds: float,
        python_coverage_config: Path,
    ) -> None:
        """Bind the explicit producer inputs without inferring a host resource grant."""
        if (
            jobs < 1
            or not math.isfinite(deadline)
            or deadline <= time.monotonic()
            or not math.isfinite(stop_seconds)
            or stop_seconds <= 0
        ):
            raise ValueError("A positive concurrency and remaining deadline are required")
        self.job = job
        self.repository = repository
        self.go = go
        self.deadline = deadline
        self.jobs = jobs
        self.environment = environment
        self.stop_seconds = stop_seconds
        self.python_coverage_config = python_coverage_config
        self.python_coverage_sha256 = validate_config(python_coverage_config, repository)
        self.source = repository / "src/sc_neurocore/accel/go"
        (job / "commands").mkdir()
        (job / "artifacts").mkdir()

    def python_script(
        self, script: Path, arguments: list[str], *, optimized: bool = False
    ) -> list[str]:
        """Trace an actual owned script using explicit configuration and real interpreter flags."""
        import sys

        return [
            sys.executable,
            "-I",
            "-B",
            *(["-O"] if optimized else []),
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(self.python_coverage_config),
            str(script),
            *arguments,
        ]

    def command(
        self,
        label: str,
        argv: list[str],
        cwd: Path,
        *,
        environment: dict[str, str] | None = None,
        outputs: tuple[Path, ...] = (),
    ) -> int:
        """Run a bounded real child, preserving its actual status before judging it.

        Parameters
        ----------
        label : str
            Unique command name, separate from every native output filename.
        argv : list of str
            Exact executable and arguments, without shell interpretation.
        cwd : pathlib.Path
            Existing working directory of this owned child.
        environment : dict, optional
            Explicit child mapping; the declared Go producer environment otherwise.
        outputs : tuple of pathlib.Path, optional
            Native outputs, all outside the command receipt namespace.

        Returns
        -------
        int
            The actual reaped process status, including a genuine nonzero refusal.

        Raises
        ------
        ValueError, TimeoutError, OSError
            Inputs collide, the deadline expires or process creation fails.
        """
        if re.fullmatch(r"[a-z][a-z0-9_-]*", label) is None:
            raise ValueError("Invalid native command label")
        receipts = (self.job / "commands").resolve()
        if any(path.resolve().is_relative_to(receipts) for path in outputs):
            raise ValueError("Native output collides with command receipt namespace")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The native coverage deadline is exhausted")
        directory = receipts / label
        directory.mkdir()
        selected = self.environment if environment is None else environment
        (directory / "command.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "cwd": str(cwd),
                    "environment_keys": sorted(selected),
                    "outputs": [str(path) for path in outputs],
                },
                indent=2,
            )
            + "\n"
        )
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        try:
            with (
                (directory / "stdout.log").open("xb") as stdout,
                (directory / "stderr.log").open("xb") as stderr,
            ):
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=selected,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    if process.poll() is None:
                        self.stop_child_group(process)
        except BaseException as error:
            self.save_receipt(directory, process, started, timed_out, repr(error))
            raise
        self.save_receipt(directory, process, started, timed_out, None)
        if timed_out:
            raise TimeoutError("Native coverage command exceeded its complete deadline")
        return self.retained_terminal_status(process)

    @staticmethod
    def retained_terminal_status(process: subprocess.Popen[bytes] | None) -> int:
        """Refuse missing or still-live process custody before returning an actual exit status."""
        if process is None or process.returncode is None:
            raise RuntimeError("No actual native process terminal was retained")
        return process.returncode

    def stop_child_group(self, process: subprocess.Popen[bytes]) -> None:
        """Stop only the still-owned command session and reap its original process."""
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=self.stop_seconds)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A reaped leader can leave another member of its owned group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=self.stop_seconds)

    @staticmethod
    def save_receipt(
        directory: Path,
        process: subprocess.Popen[bytes] | None,
        started: float,
        timed_out: bool,
        operation_error: str | None,
    ) -> None:
        """Retain genuine launch/exit facts, with no invented status for a missing child."""
        with (directory / "receipt.json").open("x") as stream:
            stream.write(
                json.dumps(
                    {
                        "pid": None if process is None else process.pid,
                        "returncode": None if process is None else process.returncode,
                        "process_reaped": process is not None and process.poll() is not None,
                        "elapsed_seconds": time.monotonic() - started,
                        "timed_out": timed_out,
                        "operation_error": operation_error,
                    },
                    indent=2,
                )
                + "\n"
            )

    def require(
        self,
        label: str,
        argv: list[str],
        cwd: Path,
        *,
        environment: dict[str, str] | None = None,
        outputs: tuple[Path, ...] = (),
    ) -> None:
        """Require actual zero while retaining every failed native command receipt."""
        status = self.command(label, argv, cwd, environment=environment, outputs=outputs)
        if status != 0:
            raise RuntimeError(f"Native command {label} refused with actual status {status}")

    def build_images(self) -> dict[str, Path]:
        """Build normal and tagged images and prove the ordinary ABI excludes diagnostics."""
        new_sources = [
            "mat/coverage_snapshot.go",
            "sc_resetting_mat/coverage_snapshot.go",
            "internal/coverageprofile/snapshot.go",
            "services/mat_simulation_coverage_test.go",
        ]
        self.require("go-format", [str(self.go.parent / "gofmt"), "-l", *new_sources], self.source)
        validate_gofmt_output((self.job / "commands/go-format/stdout.log").read_text())
        result: dict[str, Path] = {}
        for model in ("mat", "sc_resetting_mat"):
            for variant in ("normal", "tagged-uninstrumented", "instrumented"):
                output = self.job / "artifacts" / variant / model / ("lib" + model + ".so")
                output.parent.mkdir(parents=True)
                args = [str(self.go), "build", "-p", str(self.jobs), "-buildmode=c-shared"]
                if variant == "tagged-uninstrumented":
                    args.append("-tags=coverage")
                if variant == "instrumented":
                    packages = (
                        MODULE + "services",
                        MODULE + model,
                        MODULE + "internal/coverageprofile",
                    )
                    args += [
                        "-tags=coverage",
                        "-cover",
                        "-covermode=atomic",
                        "-coverpkg=" + ",".join(packages),
                    ]
                args += ["-o", str(output), "./" + model]
                label = "build-" + variant + "-" + model
                self.require(label, args, self.source, outputs=(output, output.with_suffix(".h")))
                symbols = "symbols-" + variant + "-" + model
                self.require(
                    symbols, ["/usr/bin/nm", "-D", "--defined-only", str(output)], self.source
                )
                text = (self.job / "commands" / symbols / "stdout.log").read_text()
                validate_native_exports(model, variant, text)
                result[variant + "/" + model] = output
        for tagged in (False, True):
            label = "tagged-dependencies" if tagged else "normal-dependencies"
            args = [str(self.go), "list", "-deps"]
            if tagged:
                args.append("-tags=coverage")
            self.require(label, args + ["./mat", "./sc_resetting_mat"], self.source)
            dependencies = (self.job / "commands" / label / "stdout.log").read_text().splitlines()
            validate_native_dependencies(tagged, dependencies)
            self.require(
                "tagged-vet" if tagged else "normal-vet",
                [
                    str(self.go),
                    "vet",
                    "-p",
                    str(self.jobs),
                    *(["-tags=coverage"] if tagged else []),
                    "./services",
                    "./mat",
                    "./sc_resetting_mat",
                ],
                self.source,
            )
        return result

    def test_family(self, oracle: Path) -> Path:
        """Run the complete maintained family test declarations and retain native counters."""
        filenames = (
            "mat_test.go",
            "mat_configuration_test.go",
            "sc_resetting_mat_test.go",
            "sc_resetting_mat_configuration_test.go",
            "mat_simulation_coverage_test.go",
        )
        expected = [
            name
            for filename in filenames
            for name in re.findall(
                r"^func (Test\w+)\(t \*testing.T\)",
                (self.source / "services" / filename).read_text(),
                re.MULTILINE,
            )
        ]
        validate_native_test_names(expected)
        raw = self.job / "artifacts/native-tests-raw"
        raw.mkdir()
        profile = self.job / "artifacts/native-tests.txt"
        environment = {**self.environment, "NEUROCORE_MAT_ORACLE": str(oracle)}
        remaining = max(1, math.ceil(self.deadline - time.monotonic()))
        self.require(
            "native-family-tests",
            [
                str(self.go),
                "test",
                "-p",
                str(self.jobs),
                "-count=1",
                "-vet=all",
                "-json",
                "-tags=coverage",
                "-timeout=" + str(remaining) + "s",
                "-cover",
                "-covermode=atomic",
                "-coverprofile=" + str(profile),
                "./services",
                "-run",
                "^(" + "|".join(expected) + ")$",
                "-args",
                "-test.gocoverdir=" + str(raw),
            ],
            self.source,
            environment=environment,
            outputs=(profile, raw),
        )
        lines = (self.job / "commands/native-family-tests/stdout.log").read_text().splitlines()
        events = [json.loads(line) for line in lines]
        passed = validate_native_test_events(expected, events)
        (self.job / "artifacts/native-membership.json").write_text(
            json.dumps(
                {
                    "expected": expected,
                    "actual_passed": passed,
                },
                indent=2,
            )
            + "\n"
        )
        return raw

    def profiles(self, directories: list[Path], label: str) -> Path:
        """Merge real metadata and retain complete unfiltered text and function reports."""
        if not directories:
            raise ValueError("No actual native counter directories were supplied")
        merged = self.job / "artifacts" / (label + "-merged")
        merged.mkdir()
        self.require(
            label + "-merge",
            [
                str(self.go),
                "tool",
                "covdata",
                "merge",
                "-i=" + ",".join(map(str, directories)),
                "-o=" + str(merged),
            ],
            self.source,
            outputs=(merged,),
        )
        profile = self.job / "artifacts" / (label + ".txt")
        self.require(
            label + "-textfmt",
            [
                str(self.go),
                "tool",
                "covdata",
                "textfmt",
                "-i=" + str(merged),
                "-o=" + str(profile),
            ],
            self.source,
            outputs=(profile,),
        )
        self.require(
            label + "-functions",
            [
                str(self.go),
                "tool",
                "covdata",
                "func",
                "-i=" + str(merged),
            ],
            self.source,
        )
        self.require(
            label + "-metadata",
            [
                str(self.go),
                "tool",
                "covdata",
                "debugdump",
                "-i=" + str(merged),
            ],
            self.source,
        )
        return profile
