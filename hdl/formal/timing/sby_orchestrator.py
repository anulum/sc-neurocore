# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Literal

TimingKind = Literal["latency", "deadline", "bounded_liveness"]


@dataclass(frozen=True)
class TimingProperty:
    """Bounded timing contract shared by RTL monitors and model emitters."""

    name: str
    kind: TimingKind
    trigger: str
    response: str
    bound_cycles: int
    clock: str = "clk"
    reset_n: str = "rst_n"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").isalnum():
            raise ValueError("property name must be alphanumeric or underscore")
        if self.bound_cycles < 0:
            raise ValueError("bound_cycles must be non-negative")
        if self.kind not in {"latency", "deadline", "bounded_liveness"}:
            raise ValueError(f"unsupported timing property kind: {self.kind}")
        for field_name in ("trigger", "response", "clock", "reset_n"):
            value = getattr(self, field_name)
            if not value.replace("_", "").isalnum():
                raise ValueError(f"{field_name} must be a simple signal identifier")


@dataclass(frozen=True)
class ProofResult:
    """Single formal-proof execution result with fail-closed dependency status."""

    script: str
    passed: bool
    exit_code: int
    runtime_s: float
    tool: str
    solver: str
    unavailable: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str


class TimingProofOrchestrator:
    """Run timing-property SymbiYosys proofs without hiding missing tools."""

    def __init__(
        self,
        sby_script: Path,
        *,
        repo_root: Path | None = None,
        executable: str = "sby",
        solver: str = "cvc5",
        temp_root: Path | None = None,
    ) -> None:
        self.sby_script = Path(sby_script)
        self.repo_root = (
            Path(repo_root) if repo_root is not None else self._find_repo_root(self.sby_script)
        )
        self.executable = executable
        self.solver = solver
        self.temp_root = Path(temp_root) if temp_root is not None else None

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        probe = start.resolve()
        if probe.is_file():
            probe = probe.parent
        for parent in (probe, *probe.parents):
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                return parent
        return Path.cwd()

    def unavailable_tools(self) -> tuple[str, ...]:
        missing: list[str] = []
        if shutil.which(self.executable) is None:
            missing.append(self.executable)
        if shutil.which(self.solver) is None:
            missing.append(self.solver)
        return tuple(missing)

    @staticmethod
    def _tail(text: str, max_lines: int = 80) -> str:
        return "\n".join(text.splitlines()[-max_lines:])

    def prove(self, *, timeout_s: int = 120) -> ProofResult:
        missing = self.unavailable_tools()
        if missing:
            return ProofResult(
                script=str(self.sby_script),
                passed=False,
                exit_code=127,
                runtime_s=0.0,
                tool=self.executable,
                solver=self.solver,
                unavailable=missing,
                stdout_tail="",
                stderr_tail="missing formal dependency: " + ", ".join(missing),
            )

        script_path = (
            self.sby_script if self.sby_script.is_absolute() else self.repo_root / self.sby_script
        )
        if not script_path.exists():
            raise FileNotFoundError(script_path)

        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="sc_neurocore_sby_", dir=self.temp_root) as tmp:
            work_dir = Path(tmp) / "work"
            completed = subprocess.run(
                [self.executable, "-f", "-d", str(work_dir), str(script_path)],
                cwd=self.repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        runtime_s = time.perf_counter() - start
        stdout_tail = self._tail(completed.stdout)
        stderr_tail = self._tail(completed.stderr)
        combined = f"{stdout_tail}\n{stderr_tail}"
        passed = completed.returncode == 0 and "DONE (PASS" in combined
        return ProofResult(
            script=str(script_path),
            passed=passed,
            exit_code=completed.returncode,
            runtime_s=runtime_s,
            tool=self.executable,
            solver=self.solver,
            unavailable=(),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
