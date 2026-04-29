#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run surrogate-guided SC design selection from JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sc_neurocore.optimizer import (
    SurrogateSCOptimizer,
    TargetHardwareProfile,
    load_observations,
)
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)


def load_network(path: str | Path) -> list[LayerProfile]:
    """Load layer profiles from a compact JSON network manifest."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: network manifest must be a JSON object")
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{source}: network manifest must contain a non-empty layers list")
    return [
        _layer_from_record(record, source=source, index=index)
        for index, record in enumerate(layers)
    ]


def build_target(args: argparse.Namespace) -> TargetHardwareProfile:
    """Build a target hardware profile from CLI budget arguments."""
    return TargetHardwareProfile(
        name=args.target_name,
        budget=HardwareBudget(
            max_luts=args.max_luts,
            max_power_mw=args.max_power_mw,
            max_latency_cycles=args.max_latency_cycles,
        ),
    )


def optimise_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run the optimiser and return a JSON-friendly plan."""
    network = load_network(args.network)
    observations = load_observations(args.evidence) if args.evidence else []
    optimiser = SurrogateSCOptimizer(build_target(args), observations=observations)
    report = optimiser.optimise(network)
    if report is None:
        raise RuntimeError("surrogate optimiser returned no report")
    return report_to_json(report)


def report_to_json(report: SurrogateOptimizerReport) -> dict[str, Any]:
    """Convert an optimiser report into a stable JSON document."""
    layers = [
        {"id": layer_id, **_layer_config_to_json(config)}
        for layer_id, config in sorted(report.config.items())
    ]
    return {
        "target_name": report.target_name,
        "feasible": report.feasible,
        "total_luts": report.total_luts,
        "total_power_mw": report.total_power_mw,
        "total_latency_cycles": report.total_latency_cycles,
        "mean_accuracy": report.mean_accuracy,
        "training_points": report.training_points,
        "rejected_layers": list(report.rejected_layers),
        "layers": layers,
    }


def write_plan(plan: dict[str, Any], output: str | Path | None) -> None:
    """Write the plan to a file or stdout."""
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    Path(output).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select SC compiler settings from recorded benchmark evidence"
    )
    parser.add_argument("--network", required=True, type=Path, help="JSON network manifest")
    parser.add_argument("--evidence", type=Path, help="JSON benchmark/synthesis observations")
    parser.add_argument("--out", type=Path, help="Output JSON plan path")
    parser.add_argument("--target-name", default="generic-fpga", help="Target hardware label")
    parser.add_argument("--max-luts", type=int, required=True, help="Target LUT budget")
    parser.add_argument("--max-power-mw", type=float, required=True, help="Target power budget")
    parser.add_argument(
        "--max-latency-cycles",
        type=int,
        default=0,
        help="Target latency budget; 0 disables the latency cap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        write_plan(optimise_from_args(args), args.out)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


def _layer_from_record(record: Any, *, source: Path, index: int) -> LayerProfile:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: layer {index} must be a JSON object")
    try:
        layer_id = str(record["id"])
        mac_count = int(record["mac_count"])
    except KeyError as exc:
        raise ValueError(f"{source}: layer {index} missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: layer {index} has invalid mac_count") from exc
    if not layer_id:
        raise ValueError(f"{source}: layer {index} id must be non-empty")
    if mac_count < 0:
        raise ValueError(f"{source}: layer {index} mac_count must be non-negative")
    return LayerProfile(
        id=layer_id,
        mac_count=mac_count,
        is_critical_path=bool(record.get("is_critical_path", False)),
    )


def _layer_config_to_json(config: SurrogateLayerConfig) -> dict[str, int | float | str]:
    return {
        "bitstream_length": config.bitstream_length,
        "decorrelator": config.decorrelator,
        "mode": config.mode,
        "precision_bits": config.precision_bits,
        "lfsr_polynomial": config.lfsr_polynomial,
        "luts_used": config.luts_used,
        "power_used": config.power_used,
        "latency_cycles": config.latency_cycles,
        "accuracy_score": config.accuracy_score,
        "utility_score": config.utility_score,
    }


if __name__ == "__main__":
    raise SystemExit(main())
