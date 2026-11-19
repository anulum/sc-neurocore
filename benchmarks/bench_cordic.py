# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Circular CORDIC iCE40 benchmark

"""Measure Q8.8 standalone circular units using Yosys and nextpnr."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory

from sc_neurocore.compiler import (
    cordic_latency,
    emit_cordic_verilog,
    emit_function_unit_verilog,
    function_unit_functions,
)

_OPERATIONS = ("sin", "cos", "arctan", "atan2")
_WIDTH = 16
_FRACTION = 8
_CLOCK_MHZ = 100
_ROUTE_SEED = 1
_SOURCE_PATHS = (
    "benchmarks/bench_cordic.py",
    "src/sc_neurocore/compiler/cordic.py",
    "src/sc_neurocore/compiler/cordic_verilog.py",
    "src/sc_neurocore/compiler/function_unit_hardware.py",
    "src/sc_neurocore/compiler/function_unit_ir.py",
    "src/sc_neurocore/compiler/function_unit_programs.py",
    "src/sc_neurocore/compiler/function_units.py",
    "src/sc_neurocore/compiler/expr_lut_tables.py",
    "src/sc_neurocore/compiler/verilog_expr_emitter.py",
)


@dataclass(frozen=True)
class _Design:
    """The emitted standalone top and its public interface latency."""

    lowering: str
    operation: str
    top: str
    source: str
    latency: int


def _designs() -> list[_Design]:
    """Emit each clocked unit and each supported combinational comparison."""
    designs = [
        _Design(
            "cordic",
            operation,
            "cordic_unit",
            emit_cordic_verilog(operation, data_width=_WIDTH, fraction=_FRACTION),
            cordic_latency(operation, data_width=_WIDTH, fraction=_FRACTION),
        )
        for operation in _OPERATIONS
    ]
    for lowering in ("lut-v1", "v2"):
        supported = function_unit_functions(lowering)
        designs.extend(
            _Design(
                lowering,
                operation,
                "function_unit",
                emit_function_unit_verilog(
                    operation, data_width=_WIDTH, fraction=_FRACTION, function_lowering=lowering
                ),
                0,
            )
            for operation in _OPERATIONS
            if operation in supported
        )
    return designs


def _run(command: list[str], directory: Path, name: str, timeout: int) -> None:
    """Capture the log and stop the native process group and its scratch on timeout."""
    with (
        TemporaryDirectory(prefix="native-", dir=directory) as scratch,
        (directory / name).open("wb") as log,
        subprocess.Popen(
            command,
            cwd=directory,
            env=os.environ
            | {"TERM": "xterm", "COLUMNS": "4096", "TMPDIR": str(Path(scratch).resolve())},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            raise
        subprocess.CompletedProcess(command, return_code).check_returncode()


def _route(directory: Path, nextpnr: str, timeout: int) -> dict[str, object]:
    """Measure the single clock on an unconstrained-pin UP5K SG48 design."""
    _run(
        [
            nextpnr,
            "--up5k",
            "--package",
            "sg48",
            "--pcf-allow-unconstrained",
            "--seed",
            str(_ROUTE_SEED),
            "--freq",
            str(_CLOCK_MHZ),
            "--timing-allow-fail",
            "--json",
            "mapped.json",
            "--asc",
            "mapped.asc",
            "--report",
            "routing.json",
        ],
        directory,
        "nextpnr.log",
        timeout,
    )
    raw = (directory / "routing.json").read_bytes()
    (clock,) = json.loads(raw)["fmax"].values()
    achieved = float(clock["achieved"])
    return {
        "achieved_mhz": achieved,
        "meets_target": achieved >= _CLOCK_MHZ,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _measure(design: _Design, root: Path, args: argparse.Namespace) -> dict[str, object]:
    """Measure the generated module and report any bounded v2 synthesis timeout."""
    directory = root / f"{design.lowering}-{design.operation}"
    directory.mkdir()
    source = design.source.encode("utf-8")
    (directory / "source.v").write_bytes(source)
    result: dict[str, object] = {
        "lowering": design.lowering,
        "operation": design.operation,
        "latency_edges": design.latency,
        "rtl_sha256": hashlib.sha256(source).hexdigest(),
    }
    try:
        _run(
            [
                args.yosys,
                "-Q",
                "-T",
                "-p",
                f"read_verilog source.v; synth_ice40 -device u -dsp -top {design.top} "
                "-json mapped.json; stat",
            ],
            directory,
            "yosys.log",
            args.timeout,
        )
    except subprocess.TimeoutExpired:
        if design.lowering != "v2":
            raise
        return result | {"status": "synthesis_timeout"}

    raw = (directory / "mapped.json").read_bytes()
    cells = Counter(
        cell["type"] for cell in json.loads(raw)["modules"][design.top]["cells"].values()
    )
    if not cells:
        raise ValueError(f"Yosys mapped no cells for {design.lowering}/{design.operation}")
    result.update(status="mapped", mapped_sha256=hashlib.sha256(raw).hexdigest())
    result["cells"] = dict(sorted(cells.items()))
    result["flip_flops"] = sum(count for name, count in cells.items() if name.startswith("SB_DFF"))
    if design.lowering == "cordic" and design.operation != "atan2":
        result["route"] = _route(directory, args.nextpnr, args.timeout)
    return result


def _version(command: str, flag: str) -> str:
    """Read the native tool version from either standard output stream."""
    response = subprocess.run([command, flag], check=True, capture_output=True, text=True)
    return (response.stdout + response.stderr).strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Measure all supported cases and write a source-bound JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, help="Keep raw evidence in a new directory")
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--nextpnr", default="nextpnr-ice40")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument(
        "--timeout", type=int, default=240, help="Seconds per native tool invocation"
    )
    args = parser.parse_args(argv)
    if args.timeout < 1 or (args.jobs is not None and args.jobs < 1):
        parser.error("--timeout and --jobs must be positive")

    repo = Path(__file__).resolve().parents[1]
    report: dict[str, object] = {
        "schema_version": "sc-neurocore.cordic-ice40-benchmark.v1",
        "data_width": _WIDTH,
        "fraction": _FRACTION,
        "yosys": _version(args.yosys, "-V"),
        "nextpnr": _version(args.nextpnr, "--version"),
        "mapping": "synth_ice40 -device u -dsp",
        "route": {"device": "UP5K", "package": "SG48", "seed": _ROUTE_SEED, "mhz": _CLOCK_MHZ},
        "tool_timeout_seconds": args.timeout,
        "source_sha256": {
            path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in _SOURCE_PATHS
        },
    }
    designs = _designs()
    with TemporaryDirectory(prefix="bench-cordic-") as temporary:
        directory = Path(temporary)
        if args.artifacts is not None:
            directory = args.artifacts
            directory.mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=args.jobs or len(designs)) as pool:
            results = [pool.submit(_measure, design, directory, args) for design in designs]
            report["records"] = [item.result() for item in results]
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
