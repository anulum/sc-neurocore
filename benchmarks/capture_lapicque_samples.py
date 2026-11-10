# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retain the frozen Lapicque harness's actual seven timing samples

"""Retain the original Lapicque benchmark samples through its real command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import threading
from types import CodeType, FrameType, FunctionType
from typing import cast

_BACKENDS = ("python", "rust", "julia", "go", "mojo")
_HARNESS = "benchmarks/bench_model_lapicque.py"
_SOURCE_SHA256 = "0c709afd6567d7102ef828717b9399f91204300084b58042f94644e4df1e736a"
_TOOL_ID = 5  # CPython OPTIMIZER_ID; an existing optimizer/JIT owner is never displaced.
_TOOL_NAME = "lapicque-raw-duration-return-observer"


def _sha256(path: Path) -> str:
    """Fingerprint original input, control and result bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Create a result without replacing an earlier run."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _instrumentation() -> list[str]:
    """Describe active instrumentation without exposing foreign callback identities."""
    active: list[str] = []
    for label, callback in (
        ("main-thread profiling", sys.getprofile()),
        ("main-thread tracing", sys.gettrace()),
        ("thread profiling", threading.getprofile()),
        ("thread tracing", threading.gettrace()),
    ):
        if callback is not None:
            active.append(label)
    active.extend(
        f"Python monitoring tool {tool}"
        for tool in range(6)
        if tool != _TOOL_ID and sys.monitoring.get_tool(tool) is not None
    )
    return active


def _report_aggregates(
    report: Path,
    records: dict[str, dict[str, object]],
    errors: list[str],
) -> tuple[bool, str | None]:
    """Retain report identity and reject unreadable or malformed post-run aggregates."""
    report_sha256: str | None = None
    try:
        if not report.is_file():
            errors.append("original harness report is absent")
            return False, None
        raw = report.read_bytes()
        report_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        matches = set(records) == set(_BACKENDS) and all(
            payload["backends"][backend]["used"] is True
            and payload["backends"][backend]["median_call_ms"] == records[backend]["median_call_ms"]
            and payload["backends"][backend]["minimum_call_ms"]
            == records[backend]["minimum_call_ms"]
            for backend in _BACKENDS
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"report validation failed: {type(error).__name__}: {error}")
        return False, report_sha256
    return matches, report_sha256


@dataclass(frozen=True)
class TimingFrame:
    """Retain the real returning code and local values for diagnostic validation.

    Attributes
    ----------
    code : types.CodeType
        Code object of the actual returning frame.
    values : dict of str to object
        Shallow copy of that frame's locals; no elapsed values are invented.
    """

    code: CodeType
    values: dict[str, object]

    @classmethod
    def capture(cls, frame: FrameType) -> TimingFrame:
        """Copy a genuine interpreter frame after its original measurement.

        Parameters
        ----------
        frame : types.FrameType
            Real frame supplied by the interpreter's return event.

        Returns
        -------
        TimingFrame
            Actual code identity and a shallow copy of the observed locals.
        """
        return cls(frame.f_code, dict(frame.f_locals))


@dataclass
class TimingReturns:
    """Accumulate checked original returns without attesting performance eligibility.

    Attributes
    ----------
    code : types.CodeType
        Exact original measurement code expected by this collection.
    diagnostic : bool
        Whether existing instrumentation is admitted for investigation.
    rows : list of dict
        Checked original samples and their original returned aggregates.
    errors : list of str
        Rejected event contracts; a rejection never adds a valid row.
    """

    code: CodeType
    diagnostic: bool
    rows: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def record_return(
    records: TimingReturns,
    frame: TimingFrame,
    event: tuple[CodeType, int, object],
) -> None:
    """Check and retain an original return, including diagnostic event replay.

    Parameters
    ----------
    records : TimingReturns
        Collection bound to the original code and instrumentation policy.
    frame : TimingFrame
        Snapshot captured from the genuine returning interpreter frame.
    event : tuple of types.CodeType, int and object
        Original return code, bytecode offset and returned five-field packet.

    Notes
    -----
    Real-event replay validates the recording contract. It does not run the
    benchmark, establish isolation or make a performance-admissibility claim.
    Failed contracts append an error and add no row.
    """
    code, instruction_offset, result = event
    try:
        instrumentation = _instrumentation()
        if not records.diagnostic and instrumentation:
            raise RuntimeError("foreign instrumentation became active during measurement")
        if code is not records.code or frame.code is not records.code:
            raise RuntimeError("return frame does not match the frozen measured function")
        values: dict[str, object] = frame.values
        backend = values.get("backend")
        elapsed = values.get("elapsed_ms")
        if not isinstance(backend, str) or backend not in _BACKENDS:
            raise RuntimeError("unknown measured backend")
        if any(row["backend"] == backend for row in records.rows):
            raise RuntimeError("duplicate measured backend")
        if not isinstance(elapsed, list) or len(elapsed) != 7:
            raise RuntimeError("original seven timing samples are missing")
        samples: list[float] = []
        for sample in elapsed:
            if not isinstance(sample, float) or not math.isfinite(sample) or sample <= 0:
                raise RuntimeError("invalid original elapsed sample")
            samples.append(sample)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("original result packet changed")
        if result[0] != statistics.median(samples) or result[1] != min(samples):
            raise RuntimeError("raw samples disagree with the original returned aggregates")
        records.rows.append(
            {
                "backend": backend,
                "samples_ms": samples,
                "instrumentation_at_return": instrumentation,
                "sample_count": len(samples),
                "median_call_ms": statistics.median(samples),
                "minimum_call_ms": min(samples),
                "instruction_offset": instruction_offset,
                "original_elapsed_list_copied_after_timing": True,
            }
        )
    except Exception as error:
        records.errors.append(f"{type(error).__name__}: {error}")


class _DurationCapture:
    """Own one local return event and copy the harness's existing duration list."""

    def __init__(self, code: CodeType, *, diagnostic: bool) -> None:
        self.code = code
        self.diagnostic = diagnostic
        self.instrumentation_at_install: list[str] = []
        self.instrumentation_at_close: list[str] = []
        self.records = TimingReturns(code, diagnostic)
        self.rows = self.records.rows
        self.errors = self.records.errors
        self.claimed = False
        emit = partial(record_return, self.records)
        snapshot = TimingFrame.capture
        self.callback: Callable[[CodeType, int, object], None] = lambda c, i, r: emit(
            snapshot(sys._getframe(1)), (c, i, r)
        )

    def install(self) -> None:
        """Reserve the local return event and enforce the requested instrumentation mode."""
        self.instrumentation_at_install = _instrumentation()
        if self.instrumentation_at_install and not self.diagnostic:
            raise RuntimeError(
                "existing Python instrumentation; use --diagnostic for investigation"
            )
        if sys.monitoring.get_tool(_TOOL_ID) is not None:
            raise RuntimeError("raw-duration monitoring slot is already occupied")
        sys.monitoring.use_tool_id(_TOOL_ID, _TOOL_NAME)
        self.claimed = True
        previous = sys.monitoring.register_callback(
            _TOOL_ID, sys.monitoring.events.PY_RETURN, self.callback
        )
        if previous is not None or sys.monitoring.get_events(_TOOL_ID) != 0:
            raise RuntimeError("monitoring slot was not empty")
        sys.monitoring.set_local_events(_TOOL_ID, self.code, sys.monitoring.events.PY_RETURN)

    def close(self) -> None:
        """Check final instrumentation and release only this observer's local slot."""
        self.instrumentation_at_close = _instrumentation()
        if self.instrumentation_at_close and not self.diagnostic:
            self.errors.append("foreign instrumentation active at measurement completion")
        if not self.claimed:
            return
        if sys.monitoring.get_tool(_TOOL_ID) != _TOOL_NAME:
            self.errors.append("monitoring tool ownership changed; foreign slot retained")
            self.claimed = False
            return
        try:
            if sys.monitoring.get_events(_TOOL_ID) != 0:
                self.errors.append("global monitoring was enabled during measurement")
            if (
                sys.monitoring.get_local_events(_TOOL_ID, self.code)
                != sys.monitoring.events.PY_RETURN
            ):
                self.errors.append("local return-event configuration changed during measurement")
            sys.monitoring.set_local_events(_TOOL_ID, self.code, 0)
            previous = sys.monitoring.register_callback(
                _TOOL_ID, sys.monitoring.events.PY_RETURN, None
            )
            if previous is not self.callback:
                self.errors.append("return callback changed during measurement")
        finally:
            sys.monitoring.set_events(_TOOL_ID, 0)
            sys.monitoring.free_tool_id(_TOOL_ID)
            self.claimed = False


def main(argv: list[str] | None = None) -> int:
    """Run the complete unchanged five-backend Lapicque command with raw retention.

    Parameters
    ----------
    argv : list of str or None
        Explicit qualified checkout and a previously absent result directory.

    Returns
    -------
    int
        Zero only when the original command, all five lanes and raw aggregates agree.
        Diagnostic success confirms raw retention and never qualifies performance.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Documentation: docs/guides/lapicque_timing_samples.md",
    )
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="retain full-workload samples with instrumentation; exclude them from performance comparisons",
    )
    args = parser.parse_args(argv)
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("this replay profile requires CPython 3.12; qualify other minors first")
    root = args.checkout.resolve(strict=True)
    source = root / _HARNESS
    if _sha256(source) != _SOURCE_SHA256:
        raise RuntimeError("Lapicque harness differs from the reviewed frozen source")
    output = args.output_dir.absolute()
    output.mkdir(mode=0o700)
    sys.path[:0] = [str(root / "src"), str(root), str(root / "bridge")]
    spec = importlib.util.spec_from_file_location("lapicque_original_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("original benchmark loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, expected in [("N_STEPS", 100_000), ("N_REPEATS", 7), ("BACKENDS", _BACKENDS)]:
        if getattr(module, name) != expected:
            raise RuntimeError(f"original workload changed: {name}")
    measured: object = module._measure_backend
    entrypoint: object = module.main
    if not isinstance(measured, FunctionType) or not isinstance(entrypoint, FunctionType):
        raise RuntimeError("original function objects unavailable")
    if Path(measured.__code__.co_filename).resolve() != source:
        raise RuntimeError("measurement code does not belong to the original source")
    observer_sha256 = _sha256(Path(__file__))
    capture = _DurationCapture(measured.__code__, diagnostic=args.diagnostic)
    exit_code: int | None = None
    original_exception: BaseException | None = None
    started = datetime.now(UTC).isoformat()
    try:
        capture.install()
        original_main = cast(Callable[[list[str]], int], entrypoint)
        exit_code = original_main(["--json", str(output / "original-harness-report.json")])
    except BaseException as error:
        original_exception = error
        raise
    finally:
        try:
            capture.close()
        except (RuntimeError, ValueError) as error:
            capture.errors.append(f"monitor cleanup failed: {type(error).__name__}: {error}")
        intact = False
        try:
            intact = (
                _sha256(source) == _SOURCE_SHA256
                and getattr(module, "_measure_backend", None) is measured
                and getattr(module, "main", None) is entrypoint
            )
        except OSError as error:
            capture.errors.append(f"source validation failed: {type(error).__name__}: {error}")
        report = output / "original-harness-report.json"
        records: dict[str, dict[str, object]] = {}
        for row in capture.rows:
            backend = row["backend"]
            if isinstance(backend, str):
                records[backend] = row
        aggregate_matches, report_sha256 = _report_aggregates(report, records, capture.errors)
        no_instrumentation_observed = not (
            capture.instrumentation_at_install
            or capture.instrumentation_at_close
            or any(row["instrumentation_at_return"] for row in capture.rows)
        )
        accepted = exit_code == 0 and intact and aggregate_matches and not capture.errors
        retention = {
            "schema_version": "lapicque-original-duration-observations.v1",
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "original_harness": _HARNESS,
            "original_harness_sha256": _SOURCE_SHA256,
            "observer_sha256": observer_sha256,
            "original_report_sha256": report_sha256,
            "original_main_exit_code": exit_code,
            "execution_exception_type": (
                type(original_exception).__name__ if original_exception is not None else None
            ),
            "source_and_function_identity_unchanged": intact,
            "timed_functions_replaced": False,
            "diagnostic_mode": args.diagnostic,
            "instrumentation_at_install": capture.instrumentation_at_install,
            "instrumentation_at_close": capture.instrumentation_at_close,
            "no_instrumentation_observed_at_boundaries": no_instrumentation_observed,
            "instrumentation_observation_scope": (
                "installation, each original measurement return, and completion"
            ),
            "monitoring_scope": "PY_RETURN only on the original _measure_backend code object",
            "global_monitoring_events_enabled": False,
            "rows": capture.rows,
            "observer_errors": capture.errors,
            "original_report_aggregates_match_raw_samples": aggregate_matches,
            "raw_retention_valid": accepted,
            "performance_admissible": (
                not args.diagnostic and accepted and no_instrumentation_observed
            ),
            "isolation_acceptance_requires_external_supervision": True,
            "production_performance_claim": False,
        }
        try:
            _write_json(output / "original-seven-duration-samples.json", retention)
        except OSError as retention_error:
            if original_exception is not None:
                original_exception.add_note(
                    f"raw-retention record could not be written: {retention_error}"
                )
                raise original_exception from retention_error
            raise
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
