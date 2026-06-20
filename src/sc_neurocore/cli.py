# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Minimal CLI for SC-NeuroCore

"""Minimal CLI for SC-NeuroCore."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
# CLI subprocess calls use shell-free argument vectors and bounded local tool entrypoints.
import subprocess  # nosec B404
import sys
from typing import Any, Sequence


class _OutputAction(argparse.Action):
    """Track whether ``--output`` was supplied explicitly."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        namespace.output_supplied = True


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_DEPLOY_DENSE_PARAMS = 20_000_000


def _is_valid_sha256_digest(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sc-neurocore",
        description="SC-NeuroCore — Universal Stochastic Computing Framework",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "info",
            "benchmark",
            "preflight",
            "deploy",
            "serve",
            "map-nir",
            "hub-init",
            "compile",
            "compile-nir",
            "studio",
            "studio-bootstrap-admin",
            "studio-add-browser-user",
            "collect-synthesis",
            "scnir",
            "formal",
        ],
        help="Command to run",
    )
    parser.add_argument("model", nargs="?", help="Model file (.nir) or ODE string for compile")
    parser.add_argument("scnir_path", nargs="?", help="SC-NIR JSON document path")
    parser.add_argument(
        "--target",
        default="ice40",
        choices=["ice40", "ecp5", "artix7", "zynq", "web"],
        help="Deployment target (default: ice40)",
    )
    parser.set_defaults(output_supplied=False)
    parser.add_argument(
        "--output",
        "-o",
        default="build",
        action=_OutputAction,
        help="Output directory or JSON report path for commands that emit artefacts",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help=(
            "Simulation timestep. NIR import uses this verbatim; equation "
            "compilation uses it as the dv multiplier and rejects values "
            "that quantise to 0 in Q8.8 (i.e. dt < ~0.004)."
        ),
    )
    parser.add_argument("--T", type=int, default=256, help="Bitstream length for SC layers")
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help=(
            "Expected SHA-256 digest for deploy-time .pt/.pth checkpoint loading. "
            "Required for deploy when model input is a PyTorch checkpoint."
        ),
    )
    parser.add_argument(
        "--source-kind",
        choices=["lfsr", "sobol"],
        default="lfsr",
        help="Stochastic source family for compile-nir SC-NIR source modules",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="First deterministic source seed for compile-nir SC-NIR source modules",
    )
    parser.add_argument(
        "--audit-handoff",
        action="store_true",
        help="For compile-nir, validate emitted SC-NIR HDL artefacts and write an audit report",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port for serve command")
    parser.add_argument(
        "--identity-file",
        default=None,
        help="Studio identity JSON path for Studio identity-management commands",
    )
    parser.add_argument(
        "--principal-id",
        default="svc-studio-admin",
        help="Principal for Studio identity-management commands",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Browser-login username for studio-add-browser-user",
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read browser-user password from standard input",
    )
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        default=None,
        help=(
            "Studio role grant. Repeat to grant multiple roles; "
            "studio-bootstrap-admin defaults to studio.admin and studio.viewer."
        ),
    )
    parser.add_argument(
        "--token-bytes",
        type=int,
        default=32,
        help="Entropy bytes for the generated Studio bootstrap bearer token",
    )
    parser.add_argument(
        "--expires-at-utc",
        default=None,
        help="Optional ISO-8601 UTC expiry for the Studio bootstrap identity",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow studio-bootstrap-admin to atomically replace an existing identity file",
    )
    parser.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help="Bind host for hub-init generated Studio service",
    )
    parser.add_argument(
        "--hub-image",
        default="sc-neurocore-hub:local",
        help="Container image tag used by hub-init generated Compose bundle",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="For hub-init, clear generated offline-mode environment flags",
    )
    parser.add_argument(
        "--hardware-targets",
        default="loihi2,spinnaker2,akida",
        help="Comma-separated neuromorphic targets for map-nir",
    )
    parser.add_argument(
        "--threshold", default=None, help="Threshold expression for compile (e.g. 'v > -50')"
    )
    parser.add_argument(
        "--reset", default=None, help="Reset expression for compile (e.g. 'v = -65; w = 0')"
    )
    parser.add_argument(
        "--params", default=None, help="Parameters as key=val pairs (e.g. 'E_L=-65,tau_m=10,C=1')"
    )
    parser.add_argument(
        "--init", default=None, help="Initial state as key=val pairs (e.g. 'v=-65,w=0')"
    )
    parser.add_argument("--module-name", default="sc_equation_neuron", help="Verilog module name")
    parser.add_argument(
        "--testbench", action="store_true", help="Generate testbench alongside Verilog"
    )
    parser.add_argument(
        "--synthesize", action="store_true", help="Run Yosys synthesis after compilation"
    )
    parser.add_argument("--design", help="JSON compiler-design metadata for collect-synthesis")
    parser.add_argument(
        "--utilisation",
        "--utilization",
        dest="utilisation",
        help="Vivado utilisation or Quartus fitter report for collect-synthesis",
    )
    parser.add_argument("--power", help="Vivado or Quartus power report for collect-synthesis")
    parser.add_argument("--timing", help="Optional timing report for collect-synthesis")
    parser.add_argument(
        "--accuracy-score",
        type=float,
        help="Measured model accuracy or parity score for collect-synthesis",
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
    parser.add_argument("--out", help="Output JSON evidence path for collect-synthesis")
    parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Pipeline register insertion for high-frequency targets. "
            "'auto' selects based on target frequency, or an integer N "
            "for explicit stage count. Applies to 'compile' command."
        ),
    )
    parser.add_argument(
        "--pipeline-points",
        default=None,
        help=(
            "Comma-separated list of intermediate signal names where "
            "pipeline registers should be inserted (e.g. '_mul0,_mul2'). "
            "Only used when --pipeline is not set."
        ),
    )
    parser.add_argument(
        "--adaptive-precision",
        action="store_true",
        help=(
            "Generate dual-datapath Verilog with runtime precision switching "
            "between low-precision (default Q8.8) and high-precision (default Q16.16). "
            "Applies to 'compile' command."
        ),
    )
    parser.add_argument(
        "--lp-width",
        type=int,
        default=16,
        help="Low-precision data width for adaptive precision (default: 16)",
    )
    parser.add_argument(
        "--lp-precision",
        default=None,
        help="Low-precision format for adaptive precision (e.g. Q8.8, BFP16E3X32). "
        "If unset, --lp-width/--lp-frac are used.",
    )
    parser.add_argument(
        "--lp-frac",
        type=int,
        default=8,
        help="Low-precision fractional bits for adaptive precision (default: 8)",
    )
    parser.add_argument(
        "--hp-precision",
        default=None,
        help="High-precision format for adaptive precision (e.g. Q16.16, BFP20E4X32). "
        "If unset, --hp-width/--hp-frac are used.",
    )
    parser.add_argument(
        "--hp-width",
        type=int,
        default=32,
        help="High-precision data width for adaptive precision (default: 32)",
    )
    parser.add_argument(
        "--hp-frac",
        type=int,
        default=16,
        help="High-precision fractional bits for adaptive precision (default: 16)",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=1,
        help="Input width for formal verify-network dense LIF fixtures",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=1,
        help="Output spike width for formal verify-network dense LIF fixtures",
    )
    parser.add_argument(
        "--state-width",
        type=int,
        default=16,
        help="State width for formal verify-network dense LIF fixtures",
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=0,
        help="Monitored output index for formal verify-network rate bounds",
    )
    parser.add_argument(
        "--window-cycles",
        type=int,
        default=16,
        help="Aligned rate-bound window length for formal verify-network",
    )
    parser.add_argument(
        "--max-spikes",
        type=int,
        default=1,
        help="Maximum spikes allowed inside each formal verify-network window",
    )
    parser.add_argument(
        "--refractory-cycles",
        type=int,
        default=0,
        help="Optional refractory window for formal verify-network monitored output",
    )
    parser.add_argument(
        "--antagonistic-pair",
        default=None,
        help="Optional comma-separated output pair A,B for formal verify-network exclusion",
    )
    parser.add_argument(
        "--temporal-separation",
        default=None,
        help="Optional comma-separated output pair and cycle gap A,B,CYCLES for formal verify-network",
    )
    parser.add_argument(
        "--coactivation-cap",
        type=int,
        default=None,
        help="Optional cap on simultaneously active outputs for formal verify-network",
    )
    parser.add_argument(
        "--population-silence",
        default=None,
        help=(
            "Optional post-coactivation silence contract TRIGGER_ACTIVE_OUTPUTS,SILENCE_CYCLES "
            "for formal verify-network"
        ),
    )
    parser.add_argument(
        "--population-inactivity",
        type=int,
        default=None,
        help="Optional maximum consecutive silent output cycles for formal verify-network",
    )
    parser.add_argument(
        "--spike-trace",
        default=None,
        help="Optional JSON spike trace replayed against formal verify-network rate bounds",
    )
    parser.add_argument(
        "--run-symbiyosys",
        action="store_true",
        help="Run SymbiYosys for formal verify-network when the sby executable is available",
    )
    parser.add_argument(
        "--formal-depth",
        type=int,
        default=20,
        help="SymbiYosys bounded depth for formal verify-network",
    )
    parser.add_argument(
        "--formal-mode",
        choices=["bmc", "prove", "cover"],
        default="bmc",
        help="SymbiYosys mode for formal verify-network",
    )
    args = parser.parse_args()

    if args.version:
        from sc_neurocore import __version__

        print(f"sc-neurocore {__version__}")
        return 0

    if args.command == "info":
        return _cmd_info()
    if args.command == "benchmark":
        return _cmd_benchmark()
    if args.command == "preflight":
        return _cmd_preflight()
    if args.command == "compile":
        if not args.model:
            print(
                "Error: compile requires an ODE string. Usage:\n"
                '  sc-neurocore compile "dv/dt = -(v-E_L)/tau_m + I/C" \\\n'
                '    --threshold "v > -50" --reset "v = -65" \\\n'
                '    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" \\\n'
                "    --target ice40 --testbench --synthesize"
            )
            return 1
        return _cmd_compile(args)
    if args.command == "compile-nir":
        if not args.model:
            print(
                "Error: compile-nir requires a model file. Usage:\n"
                "  sc-neurocore compile-nir model.nir --target artix7 -o build/\n"
                "  sc-neurocore compile-nir model.nir --data-width 32 --fraction 16"
            )
            return 1
        return _cmd_compile_nir(args)
    if args.command == "deploy":
        if not args.model:
            print(
                "Error: deploy requires a model file. Usage: sc-neurocore deploy model.nir --target artix7"
            )
            return 1
        return _cmd_deploy(
            args.model,
            args.target,
            args.output,
            args.dt,
            args.T,
            checkpoint_sha256=args.checkpoint_sha256,
        )
    if args.command == "serve":
        if not args.model:
            print(
                "Error: serve requires a model file. Usage: sc-neurocore serve model.nir --port 8001"
            )
            return 1
        return _cmd_serve(args.model, args.port, args.dt)
    if args.command == "map-nir":
        if not args.model:
            print(
                "Error: map-nir requires a NIR model file. Usage: sc-neurocore map-nir model.nir -o build/silicon"
            )
            return 1
        return _cmd_map_nir(args.model, args.output, args.hardware_targets, args.dt, args.T)
    if args.command == "hub-init":
        return _cmd_hub_init(
            args.output,
            args.port,
            bind_host=args.bind_host,
            image=args.hub_image,
            offline=not args.online,
        )
    if args.command == "studio":
        return _cmd_studio(args.port)
    if args.command == "studio-bootstrap-admin":
        return _cmd_studio_bootstrap_admin(args)
    if args.command == "studio-add-browser-user":
        return _cmd_studio_add_browser_user(args)
    if args.command == "collect-synthesis":
        return _cmd_collect_synthesis(args)
    if args.command == "scnir":
        return _cmd_scnir(args)
    if args.command == "formal":
        return _cmd_formal(args)

    parser.print_help()
    return 0


def _cmd_compile_nir(args: Any) -> int:
    """Compile NIR/ONNX model to Verilog RTL artefacts."""
    import os

    import nir as nir_lib

    from sc_neurocore.ir import SCNIR_HDL_HANDOFF_MANIFEST_VERSION, write_scnir
    from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork

    ext = os.path.splitext(args.model)[1].lower()
    if ext not in (".nir", ".onnx"):
        print(f"Error: compile-nir supports .nir and .onnx files, got '{ext}'")
        return 1

    data_width = getattr(args, "data_width", None) or 16
    fraction = getattr(args, "fraction", None) or 8

    print(f"[1/4] Loading model: {args.model}")
    if ext == ".nir":
        graph = nir_lib.read(args.model)
        network = from_nir(graph, dt=args.dt)
    else:
        # ONNX → NIR → SCNetwork
        graph = nir_lib.read(args.model)
        network = from_nir(graph, dt=args.dt)

    print(f"  Loaded {len(network.topo_order)} nodes")

    print("[2/4] Building NeuronGraph...")
    neuron_graph = from_scnetwork(network, dt=args.dt)
    print(f"  {neuron_graph.total_neurons} neurons, {neuron_graph.total_synapses} synapses")
    print(f"  Types: {', '.join(sorted(neuron_graph.neuron_types))}")

    print(f"[3/4] Compiling to Verilog (Q{data_width - fraction}.{fraction})...")
    result = compile_network_to_fpga(
        neuron_graph,
        module_name=args.module_name,
        data_width=data_width,
        fraction=fraction,
        bitstream_length=args.T,
        source_kind=args.source_kind,
        base_seed=args.base_seed,
        target=args.target,
    )
    print(f"  Interconnect: {result.interconnect}")
    print(f"  Neuron modules: {len(result.neuron_modules)}")
    print(f"  SC-NIR source modules: {len(result.scnir_source_modules)}")

    # Write output files
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Top module
    top_path = os.path.join(out_dir, f"{args.module_name}.v")
    with open(top_path, "w", encoding="utf-8") as f:
        f.write(result.top_module)

    # Neuron modules
    for ntype, verilog in result.neuron_modules.items():
        mod_path = os.path.join(out_dir, f"sc_nir_{ntype}.v")
        with open(mod_path, "w", encoding="utf-8") as f:
            f.write(verilog)

    # Weight ROM
    rom_path = os.path.join(out_dir, "sc_nir_weight_rom.v")
    with open(rom_path, "w", encoding="utf-8") as f:
        f.write(result.weight_rom)

    for module_name, verilog in result.scnir_source_modules.items():
        source_path = os.path.join(out_dir, f"{module_name}.v")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(verilog)

    for module_name, verilog in result.scnir_hierarchy_modules.items():
        hierarchy_path = os.path.join(out_dir, f"{module_name}.v")
        with open(hierarchy_path, "w", encoding="utf-8") as f:
            f.write(verilog)

    scnir_document_path = os.path.join(out_dir, "scnir_document.json")
    write_scnir(scnir_document_path, result.scnir_document)

    manifest_path = os.path.join(out_dir, "scnir_source_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": SCNIR_HDL_HANDOFF_MANIFEST_VERSION,
                "module_name": result.module_name,
                "bitstream_length": args.T,
                "source_kind": args.source_kind,
                "interconnect": result.interconnect,
                "q_format": result.q_format,
                "total_neurons": result.total_neurons,
                "total_synapses": result.total_synapses,
                "scnir_stream_count": len(result.scnir_document.streams),
                "scnir_signal_kinds": _scnir_signal_kind_counts(result.scnir_document),
                "scnir_signal_routes": _scnir_signal_routes(
                    result.scnir_document,
                    interconnect=result.interconnect,
                ),
                "scnir_external_inputs": [
                    entry.as_dict() for entry in result.scnir_external_inputs
                ],
                "scnir_hierarchy_instance_count": len(result.scnir_document.hierarchy),
                "scnir_hierarchy_port_count": _scnir_hierarchy_port_count(result.scnir_document),
                "sources": [entry.as_dict() for entry in result.scnir_source_manifest],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    if getattr(args, "audit_handoff", False):
        from sc_neurocore.ir import write_scnir_hdl_handoff_audit

        audit_path = os.path.join(out_dir, "scnir_handoff_audit.json")
        write_scnir_hdl_handoff_audit(out_dir, audit_path)

    print(f"[4/4] Output written to {out_dir}/")
    print(f"  {args.module_name}.v — top-level network")
    for ntype in result.neuron_modules:
        print(f"  sc_nir_{ntype}.v — {ntype} neuron module")
    print("  sc_nir_weight_rom.v — synaptic weight ROM")
    for module_name in result.scnir_source_modules:
        print(f"  {module_name}.v — SC-NIR stochastic source module")
    for module_name in result.scnir_hierarchy_modules:
        print(f"  {module_name}.v — SC-NIR hierarchy boundary module")
    print("  scnir_document.json — validated SC-NIR document")
    print("  scnir_source_manifest.json — SC-NIR source manifest")
    if getattr(args, "audit_handoff", False):
        print("  scnir_handoff_audit.json — SC-NIR HDL handoff audit")

    if result.warnings:
        print(f"\n  ⚠ {len(result.warnings)} warning(s):")
        for w in result.warnings:
            print(f"    {w}")

    return 0


def _scnir_signal_kind_counts(document: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stream in document.streams:
        signal_kind = str(stream.signal_kind)
        counts[signal_kind] = counts.get(signal_kind, 0) + 1
    return dict(sorted(counts.items()))


def _scnir_signal_routes(document: Any, *, interconnect: str) -> dict[str, str]:
    present_kinds = {str(stream.signal_kind) for stream in document.streams}
    routes = {
        "analogue_state": "direct_mac",
        "spike": "weighted_event_aer" if interconnect == "aer" else "direct_wire",
        "weight": "stochastic_source_module",
    }
    return {kind: routes[kind] for kind in routes if kind in present_kinds}


def _scnir_hierarchy_port_count(document: Any) -> int:
    return sum(len(instance.ports) for instance in document.hierarchy)


def _cmd_compile(args: Any) -> int:
    """Compile ODE equation string to Verilog RTL + optional synthesis.

    Supports three compilation modes via CLI flags:

    1. **Standard** (default): combinational datapath at the configured
       precision (``--data-width`` / ``--fraction``).
    2. **Pipelined** (``--pipeline auto|N``): insert register stages at
       multiply outputs for high-frequency targets.  ``auto`` uses
       ``critical_path_depth()`` + ``pipeline_stages_needed()`` from
       ``static_analysis.py``.  ``--pipeline-points`` selects individual
       signals to register.
    3. **Adaptive precision** (``--adaptive-precision``): generate a
       dual-datapath module with LP and HP sub-modules, hysteresis-based
       precision switching, and clock gating.  Configure LP/HP widths via
       ``--lp-width`` / ``--lp-frac`` and ``--hp-width`` / ``--hp-frac`` or
       precision strings via ``--lp-precision`` / ``--hp-precision``.
    """
    import os

    from sc_neurocore.compiler.equation_compiler import (
        generate_testbench,
    )

    # Parse params/init from comma-separated key=val strings
    def _parse_kvpairs(s: str | None) -> dict[str, float] | None:
        if not s:
            return None
        result = {}
        for pair in s.split(","):
            k, v = pair.strip().split("=")
            result[k.strip()] = float(v.strip())
        return result

    params = _parse_kvpairs(args.params)
    init = _parse_kvpairs(args.init)

    # Pipeline configuration
    pipeline_stages = 0
    pipeline_points_list = None
    if args.pipeline:
        if args.pipeline.lower() == "auto":
            # Will compute after neuron is built
            pipeline_stages = -1  # sentinel for "auto"
        else:
            pipeline_stages = int(args.pipeline)
    if args.pipeline_points and pipeline_stages <= 0:
        pipeline_points_list = [p.strip() for p in args.pipeline_points.split(",")]

    print(f"[1/4] Parsing ODE: {args.model}")

    from sc_neurocore.compiler.equation_compiler import compile_to_verilog
    from sc_neurocore.neurons.equation_builder import from_equations

    neuron = from_equations(
        args.model,
        threshold=args.threshold,
        reset=args.reset,
        params=params,
        init=init,
        dt=args.dt,
    )

    # Auto pipeline: compute from ODE critical path
    if pipeline_stages == -1:
        from sc_neurocore.compiler.static_analysis import (
            critical_path_depth,
            pipeline_stages_needed,
        )

        max_depth = max(
            (critical_path_depth(expr) for expr in neuron.equations.values()), default=0
        )
        # Default Artix-7 100 MHz — use profile if available
        freq = 100
        pipeline_stages = pipeline_stages_needed(max_depth, freq)
        print(f"  Auto-pipeline: depth={max_depth}, stages={pipeline_stages}")

    # Adaptive precision or standard compile
    if getattr(args, "adaptive_precision", False):
        from sc_neurocore.compiler.adaptive_runtime_precision import (
            compile_adaptive_precision,
        )

        verilog = compile_adaptive_precision(
            neuron,
            module_name=args.module_name,
            lp_width=args.lp_width,
            lp_frac=args.lp_frac,
            hp_width=args.hp_width,
            hp_frac=args.hp_frac,
            lp_precision=getattr(args, "lp_precision", None),
            hp_precision=getattr(args, "hp_precision", None),
        )
    else:
        verilog = compile_to_verilog(
            neuron,
            module_name=args.module_name,
            pipeline_stages=pipeline_stages,
            pipeline_points=pipeline_points_list,
        )

    print(f"  State variables: {list(neuron.equations.keys())}")
    print(f"  Parameters: {list(neuron.parameters.keys())}")

    # Write output
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    v_path = os.path.join(out_dir, f"{args.module_name}.v")
    with open(v_path, "w") as f:
        f.write(verilog)
    print(f"[2/4] Verilog written: {v_path}")

    # Testbench
    if args.testbench:
        tb_src = generate_testbench(neuron, module_name=args.module_name)
        tb_path = os.path.join(out_dir, f"tb_{args.module_name}.v")
        with open(tb_path, "w") as f:
            f.write(tb_src)
        print(f"[3/4] Testbench written: {tb_path}")
    else:
        print("[3/4] Testbench skipped (use --testbench to generate)")

    # Synthesis
    if args.synthesize:
        cfg = _TARGET_CONFIGS.get(args.target)
        if cfg and cfg["tool"] == "yosys":
            synth_ok = _auto_synthesize(out_dir, args.target, args.module_name, cfg)
            if synth_ok:
                print("[4/4] Synthesis complete")
            else:
                print("[4/4] Synthesis skipped (Yosys not found)")
        else:
            print(f"[4/4] Synthesis skipped (target '{args.target}' requires Vivado)")
    else:
        print("[4/4] Synthesis skipped (use --synthesize to run Yosys)")

    print()
    print(f"Output: {out_dir}/")
    print(f"  {args.module_name}.v — synthesizable Verilog RTL")
    if args.testbench:
        print(f"  tb_{args.module_name}.v — simulation testbench")
        print(f"  Run: iverilog -o sim {args.module_name}.v tb_{args.module_name}.v && vvp sim")
    return 0


def _cmd_serve(model_path: str, port: int, dt: float) -> int:
    """Start streaming inference server."""
    import os

    ext = os.path.splitext(model_path)[1].lower()
    if ext != ".nir":
        print(f"Error: serve currently supports .nir files only, got '{ext}'")
        return 1

    import nir as nir_lib
    from sc_neurocore.nir_bridge import from_nir
    from sc_neurocore.serve import SpikeServer

    graph = nir_lib.read(model_path)
    network = from_nir(graph, dt=dt)
    print(f"Loaded NIR graph with {len(network.topo_order)} nodes")

    server = SpikeServer(network, port=port)
    server.start(blocking=True)
    return 0


def _cmd_info() -> int:
    from sc_neurocore import __version__

    print(f"sc-neurocore {__version__}")
    print(f"Python {sys.version}")
    print(_format_engine_status(__version__))
    _print_optional_dependency_version("numpy", "NumPy")
    _print_optional_dependency_version("jax", "JAX")

    return 0


def _cmd_collect_synthesis(args: Any) -> int:
    """Collect synthesis reports into optimiser evidence JSON."""
    from sc_neurocore.optimizer import build_payload_from_reports, write_payload

    required = (
        ("design", "--design"),
        ("utilisation", "--utilisation"),
        ("power", "--power"),
        ("accuracy_score", "--accuracy-score"),
    )
    missing = [flag for attr, flag in required if getattr(args, attr) is None]
    if missing:
        joined = ", ".join(missing)
        print(f"Error: collect-synthesis requires {joined}")
        return 1

    try:
        payload = build_payload_from_reports(
            design_path=args.design,
            utilisation_path=args.utilisation,
            power_path=args.power,
            timing_path=args.timing,
            accuracy_score=args.accuracy_score,
            latency_cycles=args.latency_cycles,
            clock_mhz=args.clock_mhz,
            inferences_per_run=args.inferences_per_run,
        )
        write_payload(payload, args.out)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if args.out is not None:
        print(f"Evidence written: {args.out}")
    return 0


def _cmd_scnir(args: Any) -> int:
    """Validate or export SC-aware NIR metadata documents."""

    from pathlib import Path

    from sc_neurocore.ir import (
        SCNIRConversionConfig,
        SCNIRValidationError,
        build_scnir_compatibility_audit,
        export_scnir_from_nir,
        load_scnir,
        scnir_compatibility_matrix_dicts,
        validate_scnir_compatibility_matrix,
        upgrade_scnir_dict,
    )

    action = args.model
    path = args.scnir_path
    if action not in {
        "validate",
        "upgrade",
        "export",
        "audit-hdl",
        "compatibility",
        "closure-audit",
    } or (action not in {"compatibility", "closure-audit"} and not path):
        print("Error: usage: sc-neurocore scnir validate model.scnir.json")
        print("       or: sc-neurocore scnir upgrade model.scnir.json --output upgraded.scnir.json")
        print("       or: sc-neurocore scnir export model.nir --output model.scnir.json")
        print("       or: sc-neurocore scnir audit-hdl build/ --output scnir_audit.json")
        print("       or: sc-neurocore scnir compatibility [repo-root]")
        print("       or: sc-neurocore scnir closure-audit [repo-root] --output scnir_audit.json")
        return 1

    if action == "compatibility":
        evidence_root = Path(path) if path else Path.cwd()
        try:
            validate_scnir_compatibility_matrix(evidence_root=evidence_root)
            if getattr(args, "output_supplied", False):
                Path(args.output).write_text(
                    json.dumps(scnir_compatibility_matrix_dicts(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except (OSError, ValueError, TypeError) as exc:
            print(f"SC-NIR compatibility matrix invalid: {exc}")
            return 1

        suffix = (
            f"; report written: {args.output}" if getattr(args, "output_supplied", False) else ""
        )
        print(f"SC-NIR compatibility matrix valid: {evidence_root}{suffix}")
        return 0

    if action == "closure-audit":
        evidence_root = Path(path) if path else Path.cwd()
        try:
            closure_report = build_scnir_compatibility_audit(evidence_root=evidence_root)
            if getattr(args, "output_supplied", False):
                Path(args.output).write_text(
                    json.dumps(closure_report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except (OSError, ValueError, TypeError) as exc:
            print(f"SC-NIR closure audit invalid: {exc}")
            return 1

        suffix = (
            f"; report written: {args.output}" if getattr(args, "output_supplied", False) else ""
        )
        print(
            "SC-NIR closure audit valid: "
            f"{evidence_root} ({closure_report['primitive_count']} primitive(s), "
            f"{closure_report['audit_evidence_file_count']} evidence file(s)){suffix}"
        )
        return 0

    if action == "validate":
        try:
            document = load_scnir(path)
        except (OSError, SCNIRValidationError, ValueError) as exc:
            print(f"SC-NIR invalid: {exc}")
            return 1

        print(f"SC-NIR valid: {path} ({len(document.streams)} stream(s))")
        return 0

    if action == "upgrade":
        if not args.output:
            print("Error: scnir upgrade requires --output upgraded.scnir.json")
            return 1
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("SC-NIR document must be a JSON object")
            payload = upgrade_scnir_dict(raw)
        except (OSError, SCNIRValidationError, ValueError, TypeError) as exc:
            print(f"SC-NIR upgrade failed: {exc}")
            return 1

        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"SC-NIR upgraded: {args.output} ({len(payload['streams'])} stream(s))")
        return 0

    if action == "audit-hdl":
        from sc_neurocore.ir import SCNIRHDLHandoffAuditError, audit_scnir_hdl_handoff

        try:
            handoff_report = audit_scnir_hdl_handoff(path)
            if args.output:
                Path(args.output).write_text(
                    json.dumps(handoff_report.as_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        except (OSError, SCNIRHDLHandoffAuditError, ValueError, TypeError) as exc:
            print(f"SC-NIR HDL handoff invalid: {exc}")
            return 1

        suffix = f"; report written: {args.output}" if args.output else ""
        print(
            "SC-NIR HDL handoff valid: "
            f"{path} ({handoff_report.stream_count} stream(s), "
            f"{handoff_report.source_module_count} source module(s)){suffix}"
        )
        return 0

    if not args.output:
        print("Error: scnir export requires --output model.scnir.json")
        return 1
    try:
        document = export_scnir_from_nir(
            path,
            output_path=args.output,
            config=SCNIRConversionConfig(bitstream_length=args.T),
            dt=args.dt,
        )
    except (OSError, SCNIRValidationError, ValueError, ImportError) as exc:
        print(f"SC-NIR export failed: {exc}")
        return 1

    print(f"SC-NIR exported: {args.output} ({len(document.streams)} stream(s))")
    return 0


def _cmd_formal(args: Any) -> int:
    """Compile and replay network-level formal verification artefacts."""

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
            completed = subprocess.run(  # nosec B603
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


def _print_optional_dependency_version(module_name: str, label: str) -> None:
    loaded_module = sys.modules.get(module_name)
    if loaded_module is not None:
        version = getattr(loaded_module, "__version__", None)
        if version is not None:
            print(f"{label}: {version}")
        return
    try:
        version = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return
    print(f"{label}: {version}")


def _format_engine_status(expected_version: str) -> str:
    try:
        import sc_neurocore_engine as engine
    except ImportError:
        return "Rust engine: not available"

    version = getattr(engine, "__version__", "unknown")
    simd_tier = _safe_simd_tier(engine)
    if version != expected_version:
        return (
            f"Rust engine: {version} ({simd_tier}) [version mismatch: expected {expected_version}]"
        )
    return f"Rust engine: {version} ({simd_tier})"


def _safe_simd_tier(engine: Any) -> str:
    simd_tier = getattr(engine, "simd_tier", None)
    if not callable(simd_tier):
        return "unknown"
    try:
        return str(simd_tier())
    except Exception:
        return "unknown"


def _cmd_benchmark() -> int:
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", "pytest", "benchmarks/benchmark_suite.py", "--benchmark-only"],
    ).returncode


def _cmd_deploy(
    model_path: str,
    target: str,
    output_dir: str,
    dt: float,
    bitstream_length: int,
    *,
    checkpoint_sha256: str | None = None,
) -> int:
    """Deploy a model to FPGA or browser artefacts."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("SC-NeuroCore Deploy")
    print(f"  Model:  {model_path}")
    print(f"  Target: {target}")
    print(f"  Output: {output_dir}")
    print()

    if target == "web":
        from sc_neurocore.edge.web_deploy import WebDeploymentConfig, build_web_deployment

        try:
            manifest = build_web_deployment(
                model_path,
                output_dir,
                WebDeploymentConfig(dt=dt, bitstream_length=bitstream_length),
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

        print("[1/1] Browser deployment scaffold generated")
        print(f"  Manifest: {os.path.join(output_dir, manifest.artefacts['manifest'])}")
        print(f"  Entry:    {os.path.join(output_dir, manifest.artefacts['html'])}")
        return 0

    deployment_layer_sizes = [(1, 1)]

    # Step 1: Load model
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".nir":
        print("[1/5] Loading NIR graph...")
        import nir as nir_lib
        from sc_neurocore.nir_bridge import from_nir

        graph = nir_lib.read(model_path)
        network = from_nir(graph, dt=dt)
        print(f"  Loaded {len(network.topo_order)} nodes")
    elif ext in (".pt", ".pth"):
        print("[1/5] Loading PyTorch model and converting to SNN...")
        from sc_neurocore.security.checkpoint_loading import (
            CheckpointTrustError,
            safe_load_checkpoint,
        )
        from sc_neurocore.conversion import convert

        if not checkpoint_sha256:
            print(
                "Error: deploy requires --checkpoint-sha256 for .pt/.pth inputs "
                "(fail-closed trusted checkpoint loading)."
            )
            return 1
        if not _is_valid_sha256_digest(checkpoint_sha256):
            print("Error: --checkpoint-sha256 must be exactly 64 hexadecimal characters.")
            return 1
        trusted_sha256 = {model_path: checkpoint_sha256}
        try:
            state = safe_load_checkpoint(
                model_path,
                trusted_sha256=trusted_sha256,
                map_location="cpu",
            )
        except CheckpointTrustError as exc:
            print(f"Error: {exc}")
            return 1
        import torch

        if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
            print("Error: checkpoint must contain a state_dict-like dictionary.")
            return 1
        if not all(torch.is_tensor(v) for v in state.values()):
            print("Error: checkpoint state_dict entries must be tensors.")
            return 1

        layers: list[torch.nn.Module] = []
        weight_keys = sorted(k for k in state if k.endswith(".weight") and state[k].dim() == 2)
        if not weight_keys:
            print(
                "Error: checkpoint does not contain any 2D dense '.weight' tensors required for deploy."
            )
            return 1
        for key in weight_keys:
            weight = state[key]
            if not torch.is_floating_point(weight):
                print(f"Error: deploy weight tensor '{key}' must use floating-point dtype.")
                return 1
            if weight.shape[0] <= 0 or weight.shape[1] <= 0:
                print(f"Error: deploy weight tensor '{key}' must have non-zero 2D shape.")
                return 1
            if not torch.isfinite(weight).all().item():
                print(f"Error: deploy weight tensor '{key}' contains non-finite values.")
                return 1
        total_dense_params = sum(int(state[key].numel()) for key in weight_keys)
        if total_dense_params > _MAX_DEPLOY_DENSE_PARAMS:
            print(
                "Error: deploy checkpoint dense parameter count exceeds safety limit "
                f"({_MAX_DEPLOY_DENSE_PARAMS:,}): {total_dense_params:,}"
            )
            return 1
        deployment_layer_sizes = [
            (int(state[k].shape[1]), int(state[k].shape[0])) for k in weight_keys
        ]
        linear_layers: list[torch.nn.Linear] = []
        for idx, k in enumerate(weight_keys):
            w = state[k]
            if idx > 0:
                prev_key = weight_keys[idx - 1]
                prev_out = int(state[prev_key].shape[0])
                curr_in = int(w.shape[1])
                if curr_in != prev_out:
                    print(
                        "Error: dense deploy weights are not composition-compatible "
                        f"between '{prev_key}' (out={prev_out}) and '{k}' (in={curr_in})."
                    )
                    return 1
            linear = torch.nn.Linear(w.shape[1], w.shape[0])
            linear.weight.data.copy_(w.to(dtype=linear.weight.dtype))
            linear.bias.data.zero_()
            linear_layers.append(linear)
            layers.append(linear)
            layers.append(torch.nn.ReLU())
        if layers and isinstance(layers[-1], torch.nn.ReLU):
            layers.pop()
        model = torch.nn.Sequential(*layers)
        in_dim = linear_layers[0].in_features if linear_layers else 1
        cal_data = torch.randn(64, in_dim)
        snn = convert(model, calibration_data=cal_data, T=bitstream_length)
        network = None
        print(f"  Converted {snn.n_layers}-layer SNN, T={snn.T}")
    else:
        print(f"Error: unsupported file format '{ext}'. Supported: .nir, .pt")
        return 1

    # Step 2: Quantize weights
    print("[2/5] Quantizing weights to Q8.8...")
    from sc_neurocore.compiler.equation_compiler import Q88

    q = Q88()
    print(f"  Q8.8: {q.data_width - q.fraction} integer + {q.fraction} fraction bits")

    # Step 3: Generate Verilog
    print("[3/5] Generating SystemVerilog...")
    from sc_neurocore.compiler.equation_compiler import equation_to_fpga

    neuron, sv_code = equation_to_fpga(
        "dv/dt = (-v + I) / tau",
        threshold="v > 1.0",
        reset="v = 0.0",
        params={"tau": 20.0},
        module_name="sc_deploy_lif",
    )
    sv_path = os.path.join(output_dir, "sc_deploy_lif.sv")
    with open(sv_path, "w") as f:
        f.write(sv_code)
    print(f"  Generated {len(sv_code)} chars -> {sv_path}")

    # Step 4: Copy HDL modules
    print("[4/5] Copying HDL modules...")
    hdl_src = os.path.join(os.path.dirname(__file__), "..", "..", "hdl")
    if not os.path.isdir(hdl_src):
        hdl_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "hdl")
    hdl_dst = os.path.join(output_dir, "hdl")
    if os.path.isdir(hdl_src):
        import shutil

        if os.path.exists(hdl_dst):
            shutil.rmtree(hdl_dst)
        shutil.copytree(hdl_src, hdl_dst, ignore=shutil.ignore_patterns("tb_*", "formal"))
        n_copied = len([f for f in os.listdir(hdl_dst) if f.endswith(".v")])
        print(f"  Copied {n_copied} Verilog modules to {hdl_dst}/")
    else:
        print("  Warning: HDL source directory not found, skipping copy")

    # Step 5: Generate project files
    print("[5/5] Generating project files...")
    _generate_project(output_dir, target, "sc_deploy_lif")
    from sc_neurocore.edge.power_thermal import PowerThermalConfig, write_power_thermal_model

    power_model_path = write_power_thermal_model(
        output_dir,
        PowerThermalConfig(
            target=target,
            layer_sizes=tuple(deployment_layer_sizes),
            bitstream_length=bitstream_length,
            clock_mhz=100.0,
        ),
    )
    print(f"  Power/thermal model -> {power_model_path}")

    # Step 6: Auto-synthesize if open-source toolchain available
    cfg = _TARGET_CONFIGS[target]
    if cfg["tool"] == "yosys":
        synth_ok = _auto_synthesize(output_dir, target, "sc_deploy_lif", cfg)
    else:
        synth_ok = False

    print()
    print(f"Deploy complete. Project in {output_dir}/")
    if synth_ok:
        print("Synthesis succeeded. Results in output directory.")
    elif cfg["tool"] == "yosys":
        print("Yosys not found. To synthesize manually:")
        print(f"  cd {output_dir} && make synth")
    else:
        print("Vivado project generated. To synthesize:")
        print(f"  cd {output_dir} && vivado -mode batch -source project.tcl")
    return 0


def _cmd_map_nir(
    model_path: str,
    output_dir: str,
    hardware_targets: str,
    dt: float,
    bitstream_length: int,
) -> int:
    """Generate deterministic silicon-mapping reports for a NIR graph."""
    import os

    if os.path.splitext(model_path)[1].lower() != ".nir":
        print("Error: map-nir supports .nir files only")
        return 1

    targets = tuple(item.strip() for item in hardware_targets.split(",") if item.strip())
    if not targets:
        print("Error: --hardware-targets must name at least one target")
        return 1

    try:
        import nir as nir_lib
        from sc_neurocore.nir_bridge import from_nir
        from sc_neurocore.nir_bridge.silicon_mapping import (
            SiliconMappingConfig,
            write_silicon_mapping_report,
        )

        graph = nir_lib.read(model_path)
        network = from_nir(graph, dt=dt)
        report_path = write_silicon_mapping_report(
            output_dir,
            network,
            SiliconMappingConfig(targets=targets, bitstream_length=bitstream_length),
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("NIR silicon mapping report generated")
    print(f"  Targets:  {', '.join(targets)}")
    print(f"  Report:   {report_path}")
    return 0


def _cmd_hub_init(
    output_dir: str,
    port: int,
    bind_host: str = "127.0.0.1",
    image: str = "sc-neurocore-hub:local",
    offline: bool = True,
) -> int:
    """Generate a local self-hosted hub Compose bundle."""
    from sc_neurocore.hub import HubBundleConfig, write_hub_bundle

    try:
        paths = write_hub_bundle(
            output_dir,
            HubBundleConfig(studio_port=port, bind_host=bind_host, image=image, offline=offline),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("SC-NeuroCore hub bundle generated")
    print(f"  Directory: {output_dir}")
    print(f"  Compose:   {paths['compose']}")
    print(f"  Manifest:  {paths['manifest']}")
    return 0


def _auto_synthesize(output_dir: str, target: str, top_module: str, cfg: dict[str, str]) -> bool:
    """Run Yosys synthesis automatically if yosys is installed. Returns True on success."""
    import os
    import shutil

    yosys = shutil.which("yosys")
    if not yosys:
        return False

    print()
    print("[6/6] Running Yosys synthesis...")
    verilog_files = " ".join(
        [
            os.path.join("hdl", f)
            for f in os.listdir(os.path.join(output_dir, "hdl"))
            if f.endswith(".v")
        ]
        + [f"{top_module}.sv"]
    )
    synth_cmd = f"synth_{cfg['family']}"
    yosys_script = (
        f"read_verilog -sv {verilog_files}; "
        f"{synth_cmd} -top {top_module}; "
        f"write_json {top_module}.json; stat"
    )
    result = subprocess.run(  # nosec B603
        [yosys, "-p", yosys_script],
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if any(k in line for k in ("Number of cells", "Number of wires", "LUT", "SB_")):
                print(f"  {line.strip()}")
        print(f"  Synthesis JSON: {os.path.join(output_dir, top_module + '.json')}")

        # Try place-and-route if nextpnr available
        pnr_tool = shutil.which(f"nextpnr-{cfg['family']}")
        if pnr_tool:
            print("  Running nextpnr place-and-route...")
            pnr_result = subprocess.run(  # nosec B603
                [
                    pnr_tool,
                    f"--{cfg['device']}",
                    "--json",
                    f"{top_module}.json",
                    "--asc",
                    f"{top_module}.asc",
                    "--package",
                    cfg["package"],
                ],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if pnr_result.returncode == 0:
                print(f"  PnR succeeded: {top_module}.asc")
                # Try bitstream generation
                pack_tool = "icepack" if cfg["family"] == "ice40" else "ecppack"
                pack_bin = shutil.which(pack_tool)
                if pack_bin:
                    subprocess.run(  # nosec B603
                        [pack_bin, f"{top_module}.asc", f"{top_module}.bin"],
                        cwd=output_dir,
                        capture_output=True,
                        timeout=60,
                    )
                    bin_path = os.path.join(output_dir, f"{top_module}.bin")
                    if os.path.exists(bin_path):
                        size_kb = os.path.getsize(bin_path) / 1024
                        print(f"  Bitstream: {bin_path} ({size_kb:.1f} KB)")
            else:
                print("  PnR failed (nextpnr error). Synthesis JSON still available.")
        return True
    else:
        print("  Yosys synthesis failed:")
        for line in result.stderr.splitlines()[-5:]:
            print(f"    {line}")
        return False


_TARGET_CONFIGS = {
    "ice40": {"family": "ice40", "device": "hx8k", "package": "ct256", "tool": "yosys"},
    "ecp5": {"family": "ecp5", "device": "85k", "package": "CABGA381", "tool": "yosys"},
    "artix7": {"family": "xc7a", "device": "xc7a100t", "package": "csg324", "tool": "vivado"},
    "zynq": {"family": "xc7z", "device": "xc7z020", "package": "clg400", "tool": "vivado"},
}


def _generate_project(output_dir: str, target: str, top_module: str) -> None:
    import os

    cfg = _TARGET_CONFIGS[target]

    if cfg["tool"] == "yosys":
        makefile = f"""# SC-NeuroCore Deploy — {target} target
TOP = {top_module}
DEVICE = {cfg["device"]}

VERILOG_FILES = $(wildcard hdl/*.v) {top_module}.sv

.PHONY: synth pnr bitstream clean

synth:
\tyosys -p "read_verilog -sv $(VERILOG_FILES); synth_{cfg["family"]} -top $(TOP); write_json $(TOP).json; stat"

pnr: synth
\tnextpnr-{cfg["family"]} --{cfg["device"]} --json $(TOP).json --asc $(TOP).asc --package {cfg["package"]}

bitstream: pnr
\t{"icepack" if cfg["family"] == "ice40" else "ecppack"} $(TOP).asc $(TOP).bin

clean:
\trm -f *.json *.asc *.bin
"""
        with open(os.path.join(output_dir, "Makefile"), "w") as f:
            f.write(makefile)
        print(f"  Makefile for {target} (Yosys flow)")

    else:
        tcl = f"""# SC-NeuroCore Deploy — {target} Vivado project
create_project sc_deploy {output_dir}/vivado -part {cfg["device"]}-1{cfg["package"]}
add_files [glob hdl/*.v] {top_module}.sv
set_property top {top_module} [current_fileset]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -jobs 4
wait_on_run impl_1
"""
        with open(os.path.join(output_dir, "project.tcl"), "w") as f:
            f.write(tcl)
        print(f"  project.tcl for {target} (Vivado flow)")

    readme = f"""# SC-NeuroCore Deployment — {target}

Generated by `sc-neurocore deploy`.

## Files
- `{top_module}.sv` — Generated neuron module (Q8.8 fixed-point)
- `hdl/` — SC-NeuroCore Verilog library (encoders, synapses, layers)
- `{"Makefile" if cfg["tool"] == "yosys" else "project.tcl"}` — Build script

## Build
{"make synth" if cfg["tool"] == "yosys" else "vivado -mode batch -source project.tcl"}
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)


def _cmd_studio(port: int) -> int:
    """Launch the Visual SNN Design Studio (Equation Playground)."""
    try:
        import uvicorn
    except ImportError:
        print("Error: Studio requires FastAPI + Uvicorn.")
        print("Install with: pip install sc-neurocore[studio]")
        return 1

    from sc_neurocore.studio.app import create_app

    import webbrowser

    app = create_app()
    url = f"http://127.0.0.1:{port}"
    print(f"SC-NeuroCore Studio starting at {url}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def _cmd_studio_bootstrap_admin(args: Any) -> int:
    """Create the first local Studio service-account identity file."""
    from pathlib import Path

    from sc_neurocore.studio.platform import (
        DEFAULT_STUDIO_ADMIN_ROLES,
        bootstrap_studio_admin_identity,
    )

    if args.identity_file is None:
        print(
            "Error: studio-bootstrap-admin requires --identity-file /path/to/studio-identities.json"
        )
        return 1
    roles = tuple(args.roles) if args.roles is not None else DEFAULT_STUDIO_ADMIN_ROLES
    try:
        result = bootstrap_studio_admin_identity(
            Path(args.identity_file),
            principal_id=args.principal_id,
            roles=roles,
            token_bytes=args.token_bytes,
            expires_at_utc=args.expires_at_utc,
            overwrite=args.allow_overwrite,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    output = result.to_public_dict()
    output["bearer_token"] = result.bearer_token
    output["environment"] = f"SC_NEUROCORE_STUDIO_IDENTITY_FILE={result.identity_file_path}"
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _cmd_studio_add_browser_user(args: Any) -> int:
    """Add a persistent browser-login user to a Studio identity file."""
    from pathlib import Path

    from sc_neurocore.studio.platform import add_studio_browser_user_record

    if args.identity_file is None:
        print(
            "Error: studio-add-browser-user requires --identity-file /path/to/studio-identities.json"
        )
        return 1
    if args.username is None:
        print("Error: studio-add-browser-user requires --username <browser-user>")
        return 1
    if args.roles is None:
        print("Error: studio-add-browser-user requires at least one --role")
        return 1
    if not args.password_stdin:
        print("Error: studio-add-browser-user requires --password-stdin")
        return 1
    password = sys.stdin.readline()
    if password.endswith("\n"):
        password = password[:-1]
    try:
        record = add_studio_browser_user_record(
            Path(args.identity_file),
            username=args.username,
            principal_id=args.principal_id,
            roles=tuple(args.roles),
            password=password,
            expires_at_utc=args.expires_at_utc,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    output = {
        "browser_user": record.to_public_dict(),
        "environment": f"SC_NEUROCORE_STUDIO_IDENTITY_FILE={Path(args.identity_file).expanduser()}",
        "schema_version": "sc-neurocore.studio.identity.browser-user.add.v1",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _cmd_preflight() -> int:
    return subprocess.run(  # nosec B603
        [sys.executable, "tools/preflight.py"],
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
