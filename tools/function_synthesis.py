# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone function resource measurement command

"""Measure function hardware or report the complete source-bound comparison matrix.

Run from the source checkout with ``python -m tools.function_synthesis``.
The compiler package and tooling must originate in that checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

from sc_neurocore.compiler import function_unit_functions
from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_report import build_function_synthesis_report
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError


class _Arguments(argparse.Namespace):
    command: str
    cache: Path
    output: Path | None
    synthesis_timeout: float
    proof_timeout: float
    lock_timeout: float
    function: str
    data_width: int
    fraction: int
    lowering: str
    measure_missing: bool


def _seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("A deadline must be finite and positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, required=True, help="Private evidence/cache directory"
    )
    parser.add_argument(
        "--output", type=Path, help="Create a new JSON report; existing files are kept"
    )
    parser.add_argument("--synthesis-timeout", type=_seconds, default=240.0)
    parser.add_argument("--proof-timeout", type=_seconds, default=90.0)
    parser.add_argument("--lock-timeout", type=_seconds, default=600.0)
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure", help="Synthesize and prove one diagnostic function")
    measure.add_argument("function", choices=function_unit_functions())
    measure.add_argument("--data-width", type=int, default=16)
    measure.add_argument("--fraction", type=int, default=8)
    measure.add_argument("--lowering", choices=("v2", "lut-v1"), default="v2")
    report = commands.add_parser("report", help="Report all functions, lowerings and six formats")
    report.add_argument(
        "--measure-missing",
        action="store_true",
        help="Execute every missing mapping and proof; otherwise only read retained successes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a diagnostic or full report with explicit failure and completion status.

    Parameters
    ----------
    argv : list of str or None, optional
        Command arguments, excluding the program name. None uses sys.argv.

    Returns
    -------
    int
        Zero for an admitted supported diagnostic or complete matrix, one for
        an incomplete/refused report or unsupported diagnostic, and two for a
        setup/output error, and three for an unexpected internal failure.
        argparse also exits two for invalid arguments.

    Notes
    -----
    JSON uses no private host paths. A report creation time is distinct from
    each reused measurement's original qualification time. Existing output is
    refused before any synthesis. Only explicit --measure-missing starts new
    measurements for the full matrix. Single-case diagnostics never assert
    matrix completion.

    """
    args = _parser().parse_args(argv, namespace=_Arguments())
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError("Report output already exists")
        case = (
            SynthesisCase(args.function, args.data_width, args.fraction, args.lowering)
            if args.command == "measure"
            else None
        )
        source = Path(__file__)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        inspection = args.cache / "inspections" / uuid4().hex
        tools = SynthesisToolchain.inspect(inspection)
        cache = FunctionSynthesisCache(args.cache, tools)
        report = build_function_synthesis_report(
            cache,
            cases=(case,) if case is not None else None,
            measure_missing=case is not None or args.measure_missing,
            synthesis_timeout=args.synthesis_timeout,
            proof_timeout=args.proof_timeout,
            lock_timeout=args.lock_timeout,
        )
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
            raise ValueError("Command source changed during measurement")
        report["command_source_sha256"] = source_digest
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        return 0 if report["status"] == "complete" and (case is None or case.supported) else 1
    except (OSError, ValueError, SynthesisToolError):
        print(
            "Measurement refused; check the retained local evidence and output destination.",
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            "Measurement failed unexpectedly; check the retained local evidence.", file=sys.stderr
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
