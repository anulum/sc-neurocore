# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network formal verification command

"""Emit and replay network-level formal verification artefacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any


def add_formal_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register network formal verification.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "formal",
        help="Emit and replay formal network contracts",
        description="Generate dense-LIF RTL, SVA, SymbiYosys input, and a validated evidence report.",
    )
    parser.add_argument("model", nargs="?", help="Formal action; currently verify-network")
    parser.add_argument("--output", "-o", default="build", help="Formal artefact directory")
    parser.add_argument("--out", default=None, help="Optional report JSON path")
    parser.add_argument("--module-name", default="sc_equation_neuron")
    parser.add_argument("--input-width", type=int, default=1)
    parser.add_argument("--output-width", type=int, default=1)
    parser.add_argument("--state-width", type=int, default=16)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--window-cycles", type=int, default=16)
    parser.add_argument("--max-spikes", type=int, default=1)
    parser.add_argument("--refractory-cycles", type=int, default=0)
    parser.add_argument("--antagonistic-pair", default=None)
    parser.add_argument("--temporal-separation", default=None)
    parser.add_argument("--coactivation-cap", type=int, default=None)
    parser.add_argument("--population-silence", default=None)
    parser.add_argument("--population-inactivity", type=int, default=None)
    parser.add_argument("--spike-trace", default=None)
    parser.add_argument("--run-symbiyosys", action="store_true")
    parser.add_argument("--formal-depth", type=int, default=20)
    parser.add_argument("--formal-mode", choices=["bmc", "prove", "cover"], default="bmc")
    parser.set_defaults(handler=run_formal)


def run_formal(args: argparse.Namespace) -> int:
    """Compile and replay network-level formal verification artefacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``formal`` arguments.

    Returns
    -------
    int
        Zero when emitted contracts and requested verification pass, otherwise one.
    """
    from dataclasses import asdict
    from pathlib import Path

    from sc_neurocore.formal import (
        DenseLIFNetworkSpec,
        NetworkAntagonisticOutputExclusion,
        NetworkOutputTemporalSeparation,
        NetworkPopulationCoactivationCap,
        NetworkPopulationInactivityBound,
        NetworkPopulationSilenceAfterCoactivation,
        NetworkRefractoryInvariant,
        NetworkRateBound,
        compile_dense_lif_fixture_rtl,
        compile_network_antagonistic_exclusion_sva,
        compile_network_population_coactivation_sva,
        compile_network_population_inactivity_sva,
        compile_network_population_silence_sva,
        compile_network_rate_bound_sva,
        compile_network_refractory_sva,
        compile_network_temporal_separation_sva,
        replay_antagonistic_counterexample,
        replay_population_coactivation_counterexample,
        replay_population_inactivity_counterexample,
        replay_population_silence_counterexample,
        replay_rate_bound_counterexample,
        replay_refractory_counterexample,
        replay_temporal_separation_counterexample,
        validate_formal_network_report,
    )
    from sc_neurocore.formal.report_schema import FORMAL_NETWORK_REPORT_SCHEMA_VERSION
    from sc_neurocore.compiler.deployment import generate_sby_script

    if args.model != "verify-network":
        print("Error: usage: sc-neurocore formal verify-network --module-name dense_lif")
        return 1
    if args.formal_depth <= 0:
        print("Formal network contract invalid: formal-depth must be a positive integer")
        return 1
    if args.refractory_cycles < 0:
        print("Formal network contract invalid: refractory-cycles must be non-negative")
        return 1
    if args.coactivation_cap is not None and args.coactivation_cap < 0:
        print("Formal network contract invalid: coactivation-cap must be non-negative")
        return 1
    if args.population_inactivity is not None and args.population_inactivity <= 0:
        print("Formal network contract invalid: population-inactivity must be a positive integer")
        return 1

    try:
        network = DenseLIFNetworkSpec(
            name=args.module_name,
            input_width=args.input_width,
            output_width=args.output_width,
            state_width=args.state_width,
        )
        antagonistic_outputs = (
            _parse_antagonistic_pair(args.antagonistic_pair)
            if args.antagonistic_pair is not None
            else None
        )
        temporal_outputs = (
            _parse_temporal_separation(args.temporal_separation)
            if args.temporal_separation is not None
            else None
        )
        population_silence_values = (
            _parse_population_silence(args.population_silence)
            if args.population_silence is not None
            else None
        )
        rate_bound = NetworkRateBound(
            name=f"output{args.output_index}_rate_bound",
            output_index=args.output_index,
            window_cycles=args.window_cycles,
            max_spikes=args.max_spikes,
        )
        refractory = (
            NetworkRefractoryInvariant(
                name=f"output{args.output_index}_refractory",
                output_index=args.output_index,
                refractory_cycles=args.refractory_cycles,
            )
            if args.refractory_cycles > 0
            else None
        )
        antagonistic = (
            NetworkAntagonisticOutputExclusion(
                name=f"output{antagonistic_outputs[0]}_output{antagonistic_outputs[1]}_exclusion",
                output_a=antagonistic_outputs[0],
                output_b=antagonistic_outputs[1],
            )
            if antagonistic_outputs is not None
            else None
        )
        temporal = (
            NetworkOutputTemporalSeparation(
                name=(
                    f"output{temporal_outputs[0]}_output{temporal_outputs[1]}_temporal_separation"
                ),
                output_a=temporal_outputs[0],
                output_b=temporal_outputs[1],
                separation_cycles=temporal_outputs[2],
            )
            if temporal_outputs is not None
            else None
        )
        population = (
            NetworkPopulationCoactivationCap(
                name="population_coactivation_cap",
                max_active_outputs=args.coactivation_cap,
            )
            if args.coactivation_cap is not None
            else None
        )
        population_silence = (
            NetworkPopulationSilenceAfterCoactivation(
                name="population_silence_after_coactivation",
                trigger_active_outputs=population_silence_values[0],
                silence_cycles=population_silence_values[1],
            )
            if population_silence_values is not None
            else None
        )
        population_inactivity = (
            NetworkPopulationInactivityBound(
                name="population_inactivity_bound",
                max_silent_cycles=args.population_inactivity,
            )
            if args.population_inactivity is not None
            else None
        )
        rtl = compile_dense_lif_fixture_rtl(network)
        sva = compile_network_rate_bound_sva(network, rate_bound)
        refractory_sva = (
            compile_network_refractory_sva(network, refractory) if refractory is not None else None
        )
        antagonistic_sva = (
            compile_network_antagonistic_exclusion_sva(network, antagonistic)
            if antagonistic is not None
            else None
        )
        temporal_sva = (
            compile_network_temporal_separation_sva(network, temporal)
            if temporal is not None
            else None
        )
        population_sva = (
            compile_network_population_coactivation_sva(network, population)
            if population is not None
            else None
        )
        population_silence_sva = (
            compile_network_population_silence_sva(network, population_silence)
            if population_silence is not None
            else None
        )
        population_inactivity_sva = (
            compile_network_population_inactivity_sva(network, population_inactivity)
            if population_inactivity is not None
            else None
        )
    except ValueError as exc:
        print(f"Formal network contract invalid: {exc}")
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rtl_path = out_dir / f"{network.name}.v"
    sva_path = out_dir / f"{network.name}_rate_bound.sv"
    refractory_sva_path = out_dir / f"{network.name}_refractory.sv"
    antagonistic_sva_path = out_dir / f"{network.name}_antagonistic.sv"
    temporal_sva_path = out_dir / f"{network.name}_temporal_separation.sv"
    population_sva_path = out_dir / f"{network.name}_population_coactivation.sv"
    population_silence_sva_path = out_dir / f"{network.name}_population_silence.sv"
    population_inactivity_sva_path = out_dir / f"{network.name}_population_inactivity.sv"
    formal_bundle_path = out_dir / f"{network.name}_formal_bundle.sv"
    sby_path = out_dir / f"{network.name}.sby"
    report_path = Path(args.out) if args.out else out_dir / "formal_rate_bound_report.json"

    replay_report: dict[str, Any] | None = None
    refractory_replay_report: dict[str, Any] | None = None
    antagonistic_replay_report: dict[str, Any] | None = None
    temporal_replay_report: dict[str, Any] | None = None
    population_replay_report: dict[str, Any] | None = None
    population_silence_replay_report: dict[str, Any] | None = None
    population_inactivity_replay_report: dict[str, Any] | None = None
    replay_violated = False
    refractory_violated = False
    antagonistic_violated = False
    temporal_violated = False
    population_violated = False
    population_silence_violated = False
    population_inactivity_violated = False
    if args.spike_trace:
        try:
            trace_payload = json.loads(Path(args.spike_trace).read_text(encoding="utf-8"))
            if not isinstance(trace_payload, list):
                raise ValueError("spike trace JSON must be a list")
            replay = replay_rate_bound_counterexample(trace_payload, rate_bound)
            refractory_replay = (
                replay_refractory_counterexample(trace_payload, refractory)
                if refractory is not None
                else None
            )
            antagonistic_replay = (
                replay_antagonistic_counterexample(trace_payload, antagonistic)
                if antagonistic is not None
                else None
            )
            temporal_replay = (
                replay_temporal_separation_counterexample(trace_payload, temporal)
                if temporal is not None
                else None
            )
            population_replay = (
                replay_population_coactivation_counterexample(trace_payload, population)
                if population is not None
                else None
            )
            population_silence_replay = (
                replay_population_silence_counterexample(trace_payload, population_silence)
                if population_silence is not None
                else None
            )
            population_inactivity_replay = (
                replay_population_inactivity_counterexample(trace_payload, population_inactivity)
                if population_inactivity is not None
                else None
            )
        except (OSError, TypeError, ValueError) as exc:
            print(f"Formal replay invalid: {exc}")
            return 1
        replay_report = asdict(replay)
        replay_violated = replay.violated
        refractory_replay_report = (
            asdict(refractory_replay) if refractory_replay is not None else None
        )
        refractory_violated = bool(refractory_replay is not None and refractory_replay.violated)
        antagonistic_replay_report = (
            asdict(antagonistic_replay) if antagonistic_replay is not None else None
        )
        antagonistic_violated = bool(
            antagonistic_replay is not None and antagonistic_replay.violated
        )
        temporal_replay_report = asdict(temporal_replay) if temporal_replay is not None else None
        temporal_violated = bool(temporal_replay is not None and temporal_replay.violated)
        population_replay_report = (
            asdict(population_replay) if population_replay is not None else None
        )
        population_violated = bool(population_replay is not None and population_replay.violated)
        population_silence_replay_report = (
            asdict(population_silence_replay) if population_silence_replay is not None else None
        )
        population_silence_violated = bool(
            population_silence_replay is not None and population_silence_replay.violated
        )
        population_inactivity_replay_report = (
            asdict(population_inactivity_replay)
            if population_inactivity_replay is not None
            else None
        )
        population_inactivity_violated = bool(
            population_inactivity_replay is not None and population_inactivity_replay.violated
        )

    bundle_parts = [sva]
    if refractory_sva is not None:
        bundle_parts.append(refractory_sva)
    if antagonistic_sva is not None:
        bundle_parts.append(antagonistic_sva)
    if temporal_sva is not None:
        bundle_parts.append(temporal_sva)
    if population_sva is not None:
        bundle_parts.append(population_sva)
    if population_silence_sva is not None:
        bundle_parts.append(population_silence_sva)
    if population_inactivity_sva is not None:
        bundle_parts.append(population_inactivity_sva)
    bundle_sva = "\n".join(bundle_parts)
    sby = generate_sby_script(
        network.name,
        sva_file=formal_bundle_path.name,
        depth=args.formal_depth,
        mode=args.formal_mode,
    )
    rtl_path.write_text(rtl, encoding="utf-8")
    sva_path.write_text(sva, encoding="utf-8")
    if refractory_sva is not None:
        refractory_sva_path.write_text(refractory_sva, encoding="utf-8")
    if antagonistic_sva is not None:
        antagonistic_sva_path.write_text(antagonistic_sva, encoding="utf-8")
    if temporal_sva is not None:
        temporal_sva_path.write_text(temporal_sva, encoding="utf-8")
    if population_sva is not None:
        population_sva_path.write_text(population_sva, encoding="utf-8")
    if population_silence_sva is not None:
        population_silence_sva_path.write_text(population_silence_sva, encoding="utf-8")
    if population_inactivity_sva is not None:
        population_inactivity_sva_path.write_text(population_inactivity_sva, encoding="utf-8")
    formal_bundle_path.write_text(bundle_sva, encoding="utf-8")
    sby_path.write_text(sby, encoding="utf-8")

    symbiyosys_report: dict[str, Any] = {
        "requested": bool(args.run_symbiyosys),
        "status": "not_requested",
        "command": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "sby": str(sby_path),
    }
    if args.run_symbiyosys:
        sby_bin = shutil.which("sby")
        if sby_bin is None:
            symbiyosys_report["status"] = "tool_unavailable"
        else:
            command = [sby_bin, "-f", str(sby_path)]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            symbiyosys_report.update(
                {
                    "status": "passed" if completed.returncode == 0 else "failed",
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )

    report = {
        "schema_version": FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
        "network": asdict(network),
        "rate_bound": asdict(rate_bound),
        "refractory": asdict(refractory) if refractory is not None else None,
        "antagonistic_exclusion": asdict(antagonistic) if antagonistic is not None else None,
        "temporal_separation": asdict(temporal) if temporal is not None else None,
        "population_coactivation": asdict(population) if population is not None else None,
        "population_silence": (
            asdict(population_silence) if population_silence is not None else None
        ),
        "population_inactivity": (
            asdict(population_inactivity) if population_inactivity is not None else None
        ),
        "artifacts": {
            "rtl": str(rtl_path),
            "sva": str(sva_path),
            "rate_sva": str(sva_path),
            "refractory_sva": str(refractory_sva_path) if refractory_sva is not None else None,
            "antagonistic_sva": (
                str(antagonistic_sva_path) if antagonistic_sva is not None else None
            ),
            "temporal_sva": str(temporal_sva_path) if temporal_sva is not None else None,
            "population_sva": str(population_sva_path) if population_sva is not None else None,
            "population_silence_sva": (
                str(population_silence_sva_path) if population_silence_sva is not None else None
            ),
            "population_inactivity_sva": (
                str(population_inactivity_sva_path)
                if population_inactivity_sva is not None
                else None
            ),
            "formal_bundle": str(formal_bundle_path),
            "sby": str(sby_path),
            "report": str(report_path),
        },
        "replay": replay_report,
        "rate_replay": replay_report,
        "refractory_replay": refractory_replay_report,
        "antagonistic_replay": antagonistic_replay_report,
        "temporal_replay": temporal_replay_report,
        "population_replay": population_replay_report,
        "population_silence_replay": population_silence_replay_report,
        "population_inactivity_replay": population_inactivity_replay_report,
        "symbiyosys": symbiyosys_report,
    }
    try:
        validate_formal_network_report(report)
    except ValueError as exc:
        print(f"Formal report invalid: {exc}")
        return 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Formal network verification artifacts written: {out_dir}")
    print(f"  RTL: {rtl_path}")
    print(f"  SVA: {sva_path}")
    if refractory_sva is not None:
        print(f"  Refractory SVA: {refractory_sva_path}")
    if antagonistic_sva is not None:
        print(f"  Antagonistic SVA: {antagonistic_sva_path}")
    if temporal_sva is not None:
        print(f"  Temporal SVA: {temporal_sva_path}")
    if population_sva is not None:
        print(f"  Population SVA: {population_sva_path}")
    if population_silence_sva is not None:
        print(f"  Population silence SVA: {population_silence_sva_path}")
    if population_inactivity_sva is not None:
        print(f"  Population inactivity SVA: {population_inactivity_sva_path}")
    print(f"  Bundle: {formal_bundle_path}")
    print(f"  SBY: {sby_path}")
    print(f"  Report: {report_path}")
    if replay_report is not None:
        if replay_violated:
            print(
                "Replay violation: "
                f"cycle {replay_report['first_violation_cycle']}, "
                f"observed_spikes={replay_report['observed_spikes']}"
            )
            return 1
        print(f"Replay passed: {replay_report['cycles_checked']} cycle(s) checked")
    if refractory_replay_report is not None:
        if refractory_violated:
            print(
                "Refractory violation: "
                f"cycle {refractory_replay_report['first_violation_cycle']}, "
                f"trigger_cycle={refractory_replay_report['trigger_cycle']}"
            )
            return 1
        print(
            "Refractory replay passed: "
            f"{refractory_replay_report['cycles_checked']} cycle(s) checked"
        )
    if antagonistic_replay_report is not None:
        if antagonistic_violated:
            print(
                "Antagonistic violation: "
                f"cycle {antagonistic_replay_report['first_violation_cycle']}, "
                f"output_a={antagonistic_replay_report['output_a']}, "
                f"output_b={antagonistic_replay_report['output_b']}"
            )
            return 1
        print(
            "Antagonistic replay passed: "
            f"{antagonistic_replay_report['cycles_checked']} cycle(s) checked"
        )
    if temporal_replay_report is not None:
        if temporal_violated:
            print(
                "Temporal separation violation: "
                f"cycle {temporal_replay_report['first_violation_cycle']}, "
                f"trigger_output={temporal_replay_report['trigger_output']}, "
                f"violating_output={temporal_replay_report['violating_output']}"
            )
            return 1
        print(
            "Temporal separation replay passed: "
            f"{temporal_replay_report['cycles_checked']} cycle(s) checked"
        )
    if population_replay_report is not None:
        if population_violated:
            print(
                "Population coactivation violation: "
                f"cycle {population_replay_report['first_violation_cycle']}, "
                f"observed_active_outputs={population_replay_report['observed_active_outputs']}, "
                f"max_active_outputs={population_replay_report['max_active_outputs']}"
            )
            return 1
        print(
            "Population coactivation replay passed: "
            f"{population_replay_report['cycles_checked']} cycle(s) checked"
        )
    if population_silence_replay_report is not None:
        if population_silence_violated:
            print(
                "Population silence violation: "
                f"cycle {population_silence_replay_report['first_violation_cycle']}, "
                f"trigger_cycle={population_silence_replay_report['trigger_cycle']}, "
                "observed_active_outputs="
                f"{population_silence_replay_report['observed_active_outputs']}"
            )
            return 1
        print(
            "Population silence replay passed: "
            f"{population_silence_replay_report['cycles_checked']} cycle(s) checked"
        )
    if population_inactivity_replay_report is not None:
        if population_inactivity_violated:
            print(
                "Population inactivity violation: "
                f"cycle {population_inactivity_replay_report['first_violation_cycle']}, "
                "observed_silent_cycles="
                f"{population_inactivity_replay_report['observed_silent_cycles']}, "
                "max_silent_cycles="
                f"{population_inactivity_replay_report['max_silent_cycles']}"
            )
            return 1
        print(
            "Population inactivity replay passed: "
            f"{population_inactivity_replay_report['cycles_checked']} cycle(s) checked"
        )
    if args.run_symbiyosys:
        if symbiyosys_report["status"] == "tool_unavailable":
            print("SymbiYosys unavailable: generated .sby but skipped external proof")
        elif symbiyosys_report["status"] == "failed":
            print(f"SymbiYosys failed: returncode={symbiyosys_report['returncode']}")
            return 1
        else:
            print("SymbiYosys passed")
    return 0


def _parse_antagonistic_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise ValueError("antagonistic-pair must be two comma-separated output indexes")
    try:
        output_a, output_b = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise ValueError("antagonistic-pair must contain integer output indexes") from exc
    return output_a, output_b


def _parse_temporal_separation(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError("temporal-separation must be A,B,CYCLES")
    try:
        output_a, output_b, cycles = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise ValueError("temporal-separation must contain integer values") from exc
    if output_a < 0 or output_b < 0 or output_a == output_b or cycles <= 0:
        raise ValueError(
            "temporal-separation must contain two distinct non-negative outputs and positive cycles"
        )
    return output_a, output_b, cycles


def _parse_population_silence(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(part == "" for part in parts):
        raise ValueError("population-silence must be TRIGGER_ACTIVE_OUTPUTS,SILENCE_CYCLES")
    try:
        trigger_active_outputs, silence_cycles = (int(part, 10) for part in parts)
    except ValueError as exc:
        raise ValueError("population-silence must contain integer values") from exc
    if trigger_active_outputs <= 0 or silence_cycles <= 0:
        raise ValueError("population-silence must contain positive values")
    return trigger_active_outputs, silence_cycles
