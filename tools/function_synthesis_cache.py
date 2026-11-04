# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound successful function synthesis cache

"""Reuse exact successful mappings while retaining failed native attempts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from sc_neurocore.compiler import emit_function_unit_verilog, function_unit_hardware
from tools.function_synthesis_exports import MAPPING_FLOW
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_retention import materialize_artifacts, retain_artifacts
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError
from tools.function_synthesis_yosys import (
    FunctionSynthesisMeasurement,
    read_function_synthesis_measurement,
    run_function_synthesis,
)

_COMPILER_SOURCES = (
    "discrete_function_programs.py",
    "discrete_function_units.py",
    "exact_normal.py",
    "expr_lut_tables.py",
    "function_unit_hardware.py",
    "function_unit_ir.py",
    "function_unit_programs.py",
    "function_units.py",
    "lut_accuracy.py",
    "magnitude_function_programs.py",
    "magnitude_function_units.py",
    "power_unit_programs.py",
    "power_units.py",
    "random_unit_programs.py",
    "random_units.py",
    "reciprocal_root_programs.py",
    "reciprocal_root_units.py",
    "verilog_compiler_config.py",
    "verilog_expr_emitter.py",
)
_SOURCE_FILES = (
    *(f"src/sc_neurocore/compiler/{name}" for name in _COMPILER_SOURCES),
    "src/sc_neurocore/hdl_gen/_ident.py",
    "tools/function_synthesis_cache.py",
    "tools/function_synthesis_retention.py",
    "tools/function_synthesis_records.py",
    "tools/function_synthesis_toolchain.py",
    "tools/function_synthesis_yosys.py",
    "tools/function_synthesis_netlist.py",
    "tools/function_synthesis_polynomial.py",
    "tools/function_synthesis_word_graph.py",
    "tools/function_synthesis_word_proof.py",
    "tools/function_synthesis_structural_proof.py",
    "tools/function_synthesis_aiger.py",
    "tools/function_synthesis_dsp_proof.py",
    "tools/function_synthesis_dsp_ports.py",
    "tools/function_synthesis_dsp_contracts.py",
    "tools/function_synthesis_proof.py",
    "tools/function_synthesis_exports.py",
    "tools/function_synthesis_checkpoints.py",
    "tools/function_synthesis_bdd.py",
    "tools/function_synthesis_partition.py",
    "tools/function_synthesis_gia.py",
    "tools/function_synthesis_json.py",
    "tools/function_synthesis_source_proof.py",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _encoded(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


class FunctionSynthesisAttemptError(SynthesisToolError):
    """Retain the actual failed native attempt and its request provenance.

    Parameters
    ----------
    directory : pathlib.Path
        Private attempt container with request.json and native measurement files.
    error : SynthesisToolError
        Actual native return status and explicit elapsed-deadline evidence.

    """

    def __init__(self, directory: Path, error: SynthesisToolError) -> None:
        """Preserve native status without exposing caught exception text."""
        super().__init__(error.returncode, error.timed_out)
        self.directory = directory


@dataclass(frozen=True, slots=True)
class CachedFunctionSynthesis:
    """Describe a retained successful measurement and whether it was reused.

    Parameters
    ----------
    measurement : FunctionSynthesisMeasurement
        Actual mapped counts with complete successful historical proof.
    directory : pathlib.Path
        Exact retained source, command, netlist, statistics and proof artifacts.
    identity : str
        Content digest of the source, RTL, flow and toolchain binding.
    reused : bool
        True only when an existing successful entry was admitted again.
        Reuse does not claim that a new native mapping or proof was executed.
    recorded_at : datetime.datetime
        Original UTC qualification time, retained unchanged on reuse.

    """

    measurement: FunctionSynthesisMeasurement
    directory: Path
    identity: str
    reused: bool
    recorded_at: datetime

    @contextmanager
    def open_artifacts(self) -> Iterator[Path]:
        """Restore the exact native acquisition for inspection without retaining another copy.

        Yields
        ------
        pathlib.Path
            Original artifact filenames, complete bytes and recorded file modes.
            The temporary view is removed when the context exits.

        """
        with materialize_artifacts(self.directory) as directory:
            yield directory


class FunctionSynthesisCache:
    """Cache successful complete mappings from one frozen source checkout.

    Parameters
    ----------
    directory : pathlib.Path
        Private cache root. Unsuccessful fresh attempts remain under attempts;
        only fully admitted records enter successful.
    tools : SynthesisToolchain
        Inspected existing synthesis and proof executables and libraries.

    Notes
    -----
    Construct the cache in a fresh source-checkout process. The compiler and
    measurement tooling must originate in that same checkout. Source bindings
    are frozen at construction; edits require a fresh process, not module reload.
    Cache custody detects stale or damaged evidence, not malicious replacement
    of an entire owner-controlled store with coherently forged records.

    Raises
    ------
    OSError, ValueError
        For missing source files or compiler/tooling from different checkouts.

    """

    def __init__(self, directory: Path, tools: SynthesisToolchain) -> None:
        """Freeze owning source bindings without creating a measurement."""
        self.directory = directory.resolve()
        self.tools = tools
        self._source_root = Path(__file__).resolve().parents[1]
        compiler = self._source_root / "src/sc_neurocore/compiler/function_unit_hardware.py"
        if compiler.resolve() != Path(function_unit_hardware.__file__).resolve():
            raise ValueError("Compiler and synthesis tools must use the same source checkout")
        self._source_sha256 = self._sources()

    def _sources(self) -> dict[str, str]:
        return {name: _digest(self._source_root / name) for name in _SOURCE_FILES}

    def _check_sources(self) -> None:
        if self._sources() != self._source_sha256:
            raise ValueError("Source checkout changed after cache construction")

    @property
    def source_bindings(self) -> dict[str, str]:
        """Return independent source provenance after verifying the frozen checkout.

        Returns
        -------
        dict of str to str
            Relative owning source paths and their actual SHA-256 values.

        """
        self._check_sources()
        return dict(self._source_sha256)

    def _binding(self, case: SynthesisCase) -> dict[str, object]:
        if not case.supported:
            raise ValueError("Unsupported functions have no successful synthesis cache entry")
        self._check_sources()
        self.tools.verify()
        rtl = emit_function_unit_verilog(
            case.function,
            data_width=case.data_width,
            fraction=case.fraction,
            function_lowering=case.lowering,
        )
        return {
            "schema_version": "sc-neurocore.function-synthesis-cache.v1",
            "case": case.key,
            "source_sha256": self._source_sha256,
            "rtl_sha256": hashlib.sha256(rtl.encode("utf-8")).hexdigest(),
            "flow_sha256": hashlib.sha256(MAPPING_FLOW.encode("utf-8")).hexdigest(),
            "toolchain": self.tools.manifest(),
        }

    def _read(
        self,
        case: SynthesisCase,
        directory: Path,
        binding: dict[str, object],
        identity: str,
    ) -> CachedFunctionSynthesis:
        if directory.is_symlink() or any(
            (directory / name).is_symlink()
            for name in ("binding.json", "measurement.json", "execution.txt")
        ):
            raise ValueError("Successful cache records must not be symbolic links")
        if (directory / "binding.json").read_text(encoding="utf-8") != _encoded(binding):
            raise ValueError("Successful cache binding differs from current sources or tools")
        with materialize_artifacts(directory) as native:
            measurement = read_function_synthesis_measurement(case, native, self.tools)
        timestamp = (directory / "execution.txt").read_text(encoding="utf-8")
        recorded_at = datetime.fromisoformat(timestamp.rstrip("\n"))
        if recorded_at.tzinfo != UTC or timestamp != recorded_at.isoformat() + "\n":
            raise ValueError("Retained execution timestamp must be canonical UTC")
        expected: dict[str, object] = {
            "measurement": measurement.to_dict(),
            "execution_sha256": _digest(directory / "execution.txt"),
        }
        if (directory / "measurement.json").read_text(encoding="utf-8") != _encoded(expected):
            raise ValueError("Successful cache artifacts differ from their saved measurement")
        self._check_sources()
        return CachedFunctionSynthesis(measurement, directory, identity, True, recorded_at)

    def get(self, case: SynthesisCase) -> CachedFunctionSynthesis | None:
        """Read an atomically published success without launching or waiting for a tool.

        Parameters
        ----------
        case : SynthesisCase
            Supported canonical function and numerical format.

        Returns
        -------
        CachedFunctionSynthesis or None
            Exact admitted historical evidence, or None if no successful entry
            exists. An in-progress attempt is not a published success.

        Raises
        ------
        OSError, ValueError
            For unsupported cases, altered sources or damaged retained entries.

        """
        binding = self._binding(case)
        identity = hashlib.sha256(_encoded(binding).encode("utf-8")).hexdigest()
        entry = self.directory / "successful" / identity
        if not entry.exists() and not entry.is_symlink():
            return None
        return self._read(case, entry, binding, identity)

    def measure(
        self,
        case: SynthesisCase,
        *,
        synthesis_timeout: float = 240.0,
        proof_timeout: float = 90.0,
        lock_timeout: float = 600.0,
    ) -> CachedFunctionSynthesis:
        """Execute one fresh attempt or admit an exact successful cache entry.

        Parameters
        ----------
        case : SynthesisCase
            Supported canonical function and complete input/output geometry.
        synthesis_timeout, proof_timeout : float, optional
            Finite positive deadlines for the actual native mapping/proof calls.
        lock_timeout : float, optional
            Finite positive wait for another writer of this exact content key.

        Returns
        -------
        CachedFunctionSynthesis
            Measured values with the retained successful artifact directory.
            Existing entries require exact source/tool bindings and artifact
            hashes; no native execution is claimed for a reused result.

        Raises
        ------
        OSError, ValueError
            For invalid budgets, changed sources, damaged cache data or artifacts.
            Existing invalid entries are preserved rather than overwritten.
        filelock.Timeout
            If another same-key writer exceeds the explicit lease-wait budget.
        FunctionSynthesisAttemptError
            If a native tool fails or times out. The fresh attempt remains
            available; no successful entry is created.

        """
        for seconds in (synthesis_timeout, proof_timeout, lock_timeout):
            if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Synthesis cache deadlines must be finite and positive")
        binding = self._binding(case)
        identity = hashlib.sha256(_encoded(binding).encode("utf-8")).hexdigest()
        successful = self.directory / "successful"
        attempts = self.directory / "attempts"
        locks = self.directory / "locks"
        for path in (successful, attempts, locks):
            path.mkdir(parents=True, exist_ok=True)
        entry = successful / identity
        with FileLock(str(locks / f"{identity}.lock"), timeout=lock_timeout):
            if entry.exists() or entry.is_symlink():
                return self._read(case, entry, binding, identity)
            request = attempts / uuid4().hex
            request.mkdir()
            (request / "request.json").write_text(_encoded(binding), encoding="utf-8")
            attempt = request / "measurement"
            try:
                measurement = run_function_synthesis(
                    case,
                    attempt,
                    self.tools,
                    synthesis_timeout=synthesis_timeout,
                    proof_timeout=proof_timeout,
                )
            except SynthesisToolError as error:
                raise FunctionSynthesisAttemptError(request, error) from error
            self._check_sources()
            recorded_at = datetime.now(UTC)
            (attempt / "execution.txt").write_text(recorded_at.isoformat() + "\n", encoding="utf-8")
            record: dict[str, object] = {
                "measurement": measurement.to_dict(),
                "execution_sha256": _digest(attempt / "execution.txt"),
            }
            for name, payload in (
                ("binding.json", binding),
                ("measurement.json", record),
            ):
                with (attempt / name).open("x", encoding="utf-8") as stream:
                    stream.write(_encoded(payload))
            self._read(case, attempt, binding, identity)
            retain_artifacts(attempt)
            self._read(case, attempt, binding, identity)
            attempt.rename(entry)
            return CachedFunctionSynthesis(measurement, entry, identity, False, recorded_at)
