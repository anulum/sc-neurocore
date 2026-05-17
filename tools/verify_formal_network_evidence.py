# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI formal network evidence verifier

"""Generate and validate formal network verification evidence for CI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.formal import FormalReportValidationError, validate_formal_network_report


SCHEMA_VERSION = "sc-neurocore.formal-network-evidence-check.v0.1"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class FormalEvidenceConfig:
    """Inputs for one generated formal network evidence check."""

    output: Path
    module_name: str = "dense_lif_frontier_fixture"
    input_width: int = 3
    output_width: int = 2
    state_width: int = 16
    output_index: int = 0
    window_cycles: int = 8
    max_spikes: int = 4
    refractory_cycles: int = 2
    antagonistic_pair: str | None = "0,1"
    temporal_separation: str | None = "0,1,2"
    coactivation_cap: int | None = 1
    population_silence: str | None = "2,2"
    formal_depth: int = 20
    formal_mode: str = "bmc"

    @property
    def report_path(self) -> Path:
        """Return the generated formal report path."""
        return self.output / "formal_rate_bound_report.json"

    @property
    def manifest_path(self) -> Path:
        """Return the generated multi-output coverage manifest path."""
        return self.output / "formal_network_coverage_manifest.json"

    def output_artifact_root(self, output_index: int) -> Path:
        """Return the per-output artifact directory."""
        return self.output / f"output_{output_index}"

    def output_report_path(self, output_index: int) -> Path:
        """Return the per-output report path."""
        return self.output_artifact_root(output_index) / "formal_rate_bound_report.json"


def build_formal_verify_command(
    config: FormalEvidenceConfig,
    *,
    sby_available: bool,
    output_index: int | None = None,
) -> list[str]:
    """Build the CLI command used for the evidence check."""
    selected_output = config.output_index if output_index is None else output_index
    artifact_root = (
        config.output if output_index is None else config.output_artifact_root(output_index)
    )
    report_path = config.report_path if output_index is None else config.output_report_path(output_index)
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.cli",
        "formal",
        "verify-network",
        "--module-name",
        config.module_name,
        "--input-width",
        str(config.input_width),
        "--output-width",
        str(config.output_width),
        "--state-width",
        str(config.state_width),
        "--output-index",
        str(selected_output),
        "--window-cycles",
        str(config.window_cycles),
        "--max-spikes",
        str(config.max_spikes),
        "--refractory-cycles",
        str(config.refractory_cycles),
        "--formal-depth",
        str(config.formal_depth),
        "--formal-mode",
        config.formal_mode,
        "--output",
        str(artifact_root),
        "--out",
        str(report_path),
    ]
    if config.antagonistic_pair is not None and config.output_width >= 2:
        command.extend(["--antagonistic-pair", config.antagonistic_pair])
    if config.temporal_separation is not None and config.output_width >= 2:
        command.extend(["--temporal-separation", config.temporal_separation])
    if config.coactivation_cap is not None and config.output_width >= 2:
        command.extend(["--coactivation-cap", str(config.coactivation_cap)])
    if config.population_silence is not None and config.output_width >= 2:
        command.extend(["--population-silence", config.population_silence])
    if sby_available:
        command.append("--run-symbiyosys")
    return command


def run_formal_evidence_check(
    config: FormalEvidenceConfig,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run formal generation and validate the resulting report with artifact_root."""
    sby_available = shutil.which("sby") is not None
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "artifact_root": str(config.output),
        "coverage_manifest_path": str(config.manifest_path),
        "commands": [],
        "symbiyosys_requested": sby_available,
        "error": None,
    }
    covered_outputs: list[int] = []
    report_entries: list[dict[str, Any]] = []
    for output_index in range(config.output_width):
        command = build_formal_verify_command(
            config,
            sby_available=sby_available,
            output_index=output_index,
        )
        completed = runner(command, capture_output=True, text=True, check=False)
        command_record = {
            "output_index": output_index,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "artifact_root": str(config.output_artifact_root(output_index)),
            "report": str(config.output_report_path(output_index)),
        }
        cast_commands = summary["commands"]
        if isinstance(cast_commands, list):
            cast_commands.append(command_record)
        if completed.returncode != 0:
            summary["error"] = (
                f"output {output_index}: formal verify-network exited "
                f"with {completed.returncode}"
            )
            summary["coverage_manifest"] = _coverage_manifest(
                config,
                covered_outputs=covered_outputs,
                reports=report_entries,
            )
            return summary
        try:
            report_path = config.output_report_path(output_index)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            validate_formal_network_report(
                payload,
                artifact_root=config.output_artifact_root(output_index),
            )
            _validate_report_matches_config(payload, config)
            _validate_report_targets_output(payload, output_index=output_index)
        except (OSError, json.JSONDecodeError, FormalReportValidationError, ValueError) as exc:
            summary["error"] = f"output {output_index}: {exc}"
            summary["coverage_manifest"] = _coverage_manifest(
                config,
                covered_outputs=covered_outputs,
                reports=report_entries,
            )
            return summary

        symbiyosys = payload.get("symbiyosys", {})
        report_entries.append(
            {
                "output_index": output_index,
                "artifact_root": str(config.output_artifact_root(output_index)),
                "report": str(config.output_report_path(output_index)),
                "rate_bound": payload["rate_bound"],
                "refractory": payload["refractory"],
                "antagonistic_exclusion": payload.get("antagonistic_exclusion"),
                "temporal_separation": payload.get("temporal_separation"),
                "population_coactivation": payload.get("population_coactivation"),
                "population_silence": payload.get("population_silence"),
                "symbiyosys_status": (
                    symbiyosys.get("status") if isinstance(symbiyosys, dict) else None
                ),
            }
        )
        covered_outputs.append(output_index)

    manifest = _coverage_manifest(config, covered_outputs=covered_outputs, reports=report_entries)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["coverage_manifest"] = manifest
    summary["passed"] = True
    return summary


def _coverage_manifest(
    config: FormalEvidenceConfig,
    *,
    covered_outputs: Sequence[int],
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "sc-neurocore.formal-network-coverage.v0.1",
        "module_name": config.module_name,
        "output_width": config.output_width,
        "required_outputs": list(range(config.output_width)),
        "covered_outputs": list(covered_outputs),
        "all_outputs_covered": list(covered_outputs) == list(range(config.output_width)),
        "reports": list(reports),
    }


def _validate_report_targets_output(payload: dict[str, Any], *, output_index: int) -> None:
    rate_bound = payload.get("rate_bound")
    if not isinstance(rate_bound, dict) or rate_bound.get("output_index") != output_index:
        raise FormalReportValidationError("rate_bound.output_index does not match manifest output")
    refractory = payload.get("refractory")
    if refractory is not None:
        if not isinstance(refractory, dict) or refractory.get("output_index") != output_index:
            raise FormalReportValidationError(
                "refractory.output_index does not match manifest output"
            )


def _validate_report_matches_config(
    payload: dict[str, Any],
    config: FormalEvidenceConfig,
) -> None:
    network = payload.get("network")
    if not isinstance(network, dict):
        raise FormalReportValidationError("network must be an object")
    expected_network = {
        "name": config.module_name,
        "input_width": config.input_width,
        "output_width": config.output_width,
        "state_width": config.state_width,
    }
    for key, expected in expected_network.items():
        if network.get(key) != expected:
            raise FormalReportValidationError(
                f"network.{key} does not match manifest {key}"
            )

    expected_pair = _parse_antagonistic_pair(config.antagonistic_pair)
    antagonistic = payload.get("antagonistic_exclusion")
    if expected_pair is None:
        if antagonistic is not None:
            raise FormalReportValidationError(
                "antagonistic_exclusion must be null when manifest has no antagonistic_pair"
            )
    elif not isinstance(antagonistic, dict):
        raise FormalReportValidationError(
            "antagonistic_exclusion must be present when manifest has antagonistic_pair"
        )
    else:
        output_a, output_b = expected_pair
        if output_a >= config.output_width or output_b >= config.output_width:
            raise FormalReportValidationError(
                "antagonistic_pair references output outside manifest output_width"
            )
        if antagonistic.get("output_a") != output_a or antagonistic.get("output_b") != output_b:
            raise FormalReportValidationError(
                "antagonistic_exclusion does not match manifest antagonistic_pair"
            )

    expected_temporal = _parse_temporal_separation(config.temporal_separation)
    temporal = payload.get("temporal_separation")
    if expected_temporal is None:
        if temporal is not None:
            raise FormalReportValidationError(
                "temporal_separation must be null when manifest has no temporal_separation"
            )
    elif not isinstance(temporal, dict):
        raise FormalReportValidationError(
            "temporal_separation must be present when manifest has temporal_separation"
        )
    else:
        temporal_a, temporal_b, cycles = expected_temporal
        if temporal_a >= config.output_width or temporal_b >= config.output_width:
            raise FormalReportValidationError(
                "temporal_separation references output outside manifest output_width"
            )
        if (
            temporal.get("output_a") != temporal_a
            or temporal.get("output_b") != temporal_b
            or temporal.get("separation_cycles") != cycles
        ):
            raise FormalReportValidationError(
                "temporal_separation does not match manifest temporal_separation"
            )

    population = payload.get("population_coactivation")
    if config.coactivation_cap is None or config.output_width < 2:
        if population is not None:
            raise FormalReportValidationError(
                "population_coactivation must be null when manifest has no coactivation_cap"
            )
    elif config.coactivation_cap < 0:
        raise FormalReportValidationError("coactivation_cap must be non-negative")
    elif config.coactivation_cap > config.output_width:
        raise FormalReportValidationError(
            "coactivation_cap references more outputs than manifest output_width"
        )
    elif not isinstance(population, dict):
        raise FormalReportValidationError(
            "population_coactivation must be present when manifest has coactivation_cap"
        )
    elif population.get("max_active_outputs") != config.coactivation_cap:
        raise FormalReportValidationError(
            "population_coactivation does not match manifest coactivation_cap"
        )

    expected_silence = _parse_population_silence(config.population_silence)
    population_silence = payload.get("population_silence")
    if expected_silence is None or config.output_width < 2:
        if population_silence is not None:
            raise FormalReportValidationError(
                "population_silence must be null when manifest has no population_silence"
            )
        return
    trigger_active_outputs, silence_cycles = expected_silence
    if trigger_active_outputs > config.output_width:
        raise FormalReportValidationError(
            "population_silence references more outputs than manifest output_width"
        )
    if not isinstance(population_silence, dict):
        raise FormalReportValidationError(
            "population_silence must be present when manifest has population_silence"
        )
    if (
        population_silence.get("trigger_active_outputs") != trigger_active_outputs
        or population_silence.get("silence_cycles") != silence_cycles
    ):
        raise FormalReportValidationError(
            "population_silence does not match manifest population_silence"
        )


def _parse_antagonistic_pair(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise FormalReportValidationError(
            "antagonistic_pair must be two comma-separated output indexes"
        )
    try:
        output_a, output_b = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise FormalReportValidationError(
            "antagonistic_pair must contain integer output indexes"
        ) from exc
    if output_a < 0 or output_b < 0 or output_a == output_b:
        raise FormalReportValidationError(
            "antagonistic_pair must contain two distinct non-negative output indexes"
        )
    return output_a, output_b


def _parse_temporal_separation(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise FormalReportValidationError("temporal_separation must be A,B,CYCLES")
    try:
        output_a, output_b, cycles = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise FormalReportValidationError(
            "temporal_separation must contain integer values"
        ) from exc
    if output_a < 0 or output_b < 0 or output_a == output_b or cycles <= 0:
        raise FormalReportValidationError(
            "temporal_separation must contain two distinct non-negative outputs and positive cycles"
        )
    return output_a, output_b, cycles


def _parse_population_silence(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise FormalReportValidationError("population_silence must be TRIGGER,SILENCE_CYCLES")
    try:
        trigger_active_outputs, silence_cycles = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise FormalReportValidationError(
            "population_silence must contain integer values"
        ) from exc
    if trigger_active_outputs <= 0 or silence_cycles <= 0:
        raise FormalReportValidationError(
            "population_silence must contain positive trigger and silence values"
        )
    return trigger_active_outputs, silence_cycles


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = subprocess.run,
) -> int:
    """Run the formal evidence verifier."""
    parser = argparse.ArgumentParser(
        description="Generate and validate SC-NeuroCore formal network evidence"
    )
    parser.add_argument("--output", type=Path, default=Path("build/formal-evidence"))
    parser.add_argument("--summary", type=Path, help="Write JSON check summary")
    parser.add_argument("--module-name", default="dense_lif_frontier_fixture")
    parser.add_argument("--input-width", type=int, default=3)
    parser.add_argument("--output-width", type=int, default=2)
    parser.add_argument("--state-width", type=int, default=16)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--window-cycles", type=int, default=8)
    parser.add_argument("--max-spikes", type=int, default=4)
    parser.add_argument("--refractory-cycles", type=int, default=2)
    parser.add_argument("--antagonistic-pair", default=None)
    parser.add_argument("--temporal-separation", default=None)
    parser.add_argument("--coactivation-cap", type=int, default=None)
    parser.add_argument("--population-silence", default=None)
    parser.add_argument("--formal-depth", type=int, default=20)
    parser.add_argument("--formal-mode", choices=["bmc", "prove", "cover"], default="bmc")
    args = parser.parse_args(argv)

    config = FormalEvidenceConfig(
        output=args.output,
        module_name=args.module_name,
        input_width=args.input_width,
        output_width=args.output_width,
        state_width=args.state_width,
        output_index=args.output_index,
        window_cycles=args.window_cycles,
        max_spikes=args.max_spikes,
        refractory_cycles=args.refractory_cycles,
        antagonistic_pair=(
            args.antagonistic_pair
            if args.antagonistic_pair is not None
            else ("0,1" if args.output_width >= 2 else None)
        ),
        temporal_separation=(
            args.temporal_separation
            if args.temporal_separation is not None
            else ("0,1,2" if args.output_width >= 2 else None)
        ),
        coactivation_cap=(
            args.coactivation_cap
            if args.coactivation_cap is not None
            else (1 if args.output_width >= 2 else None)
        ),
        population_silence=(
            args.population_silence
            if args.population_silence is not None
            else ("2,2" if args.output_width >= 2 else None)
        ),
        formal_depth=args.formal_depth,
        formal_mode=args.formal_mode,
    )
    summary = run_formal_evidence_check(config, runner=runner)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
