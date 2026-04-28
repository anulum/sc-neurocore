#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Collect FPGA synthesis and power reports into optimiser evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sc_neurocore.optimizer import load_synthesis_observation
from sc_neurocore.optimizer.surrogate_sc_optimizer import BenchmarkObservation


def load_design(path: str | Path) -> dict[str, Any]:
    """Load explicit compiler-design metadata for one synthesis observation."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: design metadata must be a JSON object")
    return payload


def observation_to_record(observation: BenchmarkObservation) -> dict[str, int | float | str | bool]:
    """Convert an optimiser observation into the stable evidence JSON shape."""
    return {
        "mac_count": observation.mac_count,
        "bitstream_length": observation.bitstream_length,
        "decorrelator": observation.decorrelator,
        "mode": observation.mode,
        "precision_bits": observation.precision_bits,
        "lfsr_polynomial": observation.lfsr_polynomial,
        "luts_used": observation.luts_used,
        "power_mw": observation.power_mw,
        "latency_cycles": observation.latency_cycles,
        "accuracy_score": observation.accuracy_score,
        "is_critical_path": observation.is_critical_path,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build an evidence payload from report files and explicit metadata."""
    reports: dict[str, Path] = {"utilisation": args.utilisation, "power": args.power}
    if args.timing is not None:
        reports["timing"] = args.timing

    observation = load_synthesis_observation(
        reports,
        design=load_design(args.design),
        accuracy_score=args.accuracy_score,
        latency_cycles=args.latency_cycles,
    )
    payload: dict[str, Any] = {
        "source_reports": {name: str(path) for name, path in reports.items()},
        "observations": [observation_to_record(observation)],
    }
    energy = energy_payload(
        observation,
        clock_mhz=args.clock_mhz,
        inferences_per_run=args.inferences_per_run,
    )
    if energy is not None:
        payload["energy"] = energy
    return payload


def energy_payload(
    observation: BenchmarkObservation,
    *,
    clock_mhz: float | None,
    inferences_per_run: int | None,
) -> dict[str, float | int] | None:
    """Compute report-derived energy only when workload metadata is explicit."""
    if clock_mhz is None and inferences_per_run is None:
        return None
    if clock_mhz is None or inferences_per_run is None:
        raise ValueError("energy requires both --clock-mhz and --inferences-per-run")
    if clock_mhz <= 0.0:
        raise ValueError("clock_mhz must be positive")
    if inferences_per_run <= 0:
        raise ValueError("inferences_per_run must be positive")

    latency_seconds = observation.latency_cycles / (clock_mhz * 1_000_000.0)
    total_energy_j = (observation.power_mw / 1000.0) * latency_seconds
    return {
        "clock_mhz": clock_mhz,
        "inferences_per_run": inferences_per_run,
        "latency_seconds": latency_seconds,
        "total_energy_uj": total_energy_j * 1_000_000.0,
        "energy_uj_per_inference": (total_energy_j / inferences_per_run) * 1_000_000.0,
    }


def write_payload(payload: dict[str, Any], output: str | Path | None) -> None:
    """Write evidence JSON to a file or stdout."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Collect FPGA synthesis and power reports into optimiser evidence JSON"
    )
    parser.add_argument("--design", required=True, type=Path, help="JSON compiler-design metadata")
    parser.add_argument(
        "--utilisation",
        "--utilization",
        dest="utilisation",
        required=True,
        type=Path,
        help="Vivado utilisation or Quartus fitter report",
    )
    parser.add_argument("--power", required=True, type=Path, help="Vivado or Quartus power report")
    parser.add_argument("--timing", type=Path, help="Optional timing report carrying latency")
    parser.add_argument(
        "--accuracy-score",
        required=True,
        type=float,
        help="Measured model accuracy or parity score for this design",
    )
    parser.add_argument(
        "--latency-cycles",
        type=int,
        help="Explicit latency cycles when reports do not carry latency",
    )
    parser.add_argument("--clock-mhz", type=float, help="Clock used for energy calculation")
    parser.add_argument(
        "--inferences-per-run",
        type=int,
        help="Number of inferences represented by the reported latency",
    )
    parser.add_argument("--out", type=Path, help="Output JSON evidence path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the evidence collector."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        write_payload(build_payload(args), args.out)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
