# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation and NIR compilation commands

"""Compile equations and NIR graphs into synthesisable hardware artefacts."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .deploy import TARGET_CONFIGS, run_auto_synthesis


def add_compile_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register equation and NIR compilation commands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    equation = subparsers.add_parser(
        "compile",
        help="Compile an ODE equation into Verilog and optional HLS C++",
        description="Lower one differential equation into synthesisable hardware sources.",
    )
    equation.add_argument("model", nargs="?", help="ODE equation string")
    _add_target_argument(equation)
    _add_output_argument(equation)
    equation.add_argument("--dt", type=float, default=1.0, help="Equation timestep")
    equation.add_argument("--threshold", default=None, help="Spike threshold expression")
    equation.add_argument("--reset", default=None, help="Reset expression")
    equation.add_argument("--params", default=None, help="Comma-separated parameter assignments")
    equation.add_argument("--init", default=None, help="Comma-separated initial-state assignments")
    equation.add_argument("--module-name", default="sc_equation_neuron")
    equation.add_argument("--testbench", action="store_true")
    equation.add_argument("--synthesize", action="store_true")
    equation.add_argument("--emit-hls", action="store_true")
    equation.add_argument("--hls-tool", choices=["vitis", "catapult"], default="vitis")
    equation.add_argument("--hls-threshold", type=float, default=1.0)
    equation.add_argument("--pipeline", default=None)
    equation.add_argument("--pipeline-points", default=None)
    equation.add_argument("--adaptive-precision", action="store_true")
    equation.add_argument("--lp-width", type=int, default=16)
    equation.add_argument("--lp-precision", default=None)
    equation.add_argument("--lp-frac", type=int, default=8)
    equation.add_argument("--hp-precision", default=None)
    equation.add_argument("--hp-width", type=int, default=32)
    equation.add_argument("--hp-frac", type=int, default=16)
    equation.set_defaults(handler=run_compile)

    nir = subparsers.add_parser(
        "compile-nir",
        help="Compile a NIR or ONNX graph into a Verilog network bundle",
        description="Lower one imported network into RTL, SC-NIR metadata, and source manifests.",
    )
    nir.add_argument("model", nargs="?", help="NIR or ONNX model file")
    _add_target_argument(nir)
    _add_output_argument(nir)
    nir.add_argument("--dt", type=float, default=1.0, help="NIR simulation timestep")
    nir.add_argument("--T", type=int, default=256, help="Stochastic bitstream length")
    nir.add_argument("--data-width", type=int, default=16, help="Fixed-point word width")
    nir.add_argument("--fraction", type=int, default=8, help="Fixed-point fractional bits")
    nir.add_argument("--module-name", default="sc_equation_neuron")
    nir.add_argument("--source-kind", choices=["lfsr", "sobol"], default="lfsr")
    nir.add_argument("--base-seed", type=int, default=1)
    nir.add_argument("--audit-handoff", action="store_true")
    nir.add_argument("--interconnect", choices=["auto", "direct", "folded"], default="auto")
    nir.set_defaults(handler=run_compile_nir)


def _add_target_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        default="ice40",
        choices=["ice40", "ecp5", "artix7", "zynq", "web"],
    )


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", "-o", default="build", help="Artefact output directory")


def run_compile_nir(args: argparse.Namespace) -> int:
    """Compile a NIR or ONNX model to Verilog RTL artefacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``compile-nir`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid command input.
    """
    if not args.model:
        print(
            "Error: compile-nir requires a model file. Usage:\n"
            "  sc-neurocore compile-nir model.nir --target artix7 -o build/\n"
            "  sc-neurocore compile-nir model.nir --data-width 32 --fraction 16"
        )
        return 1
    if args.data_width <= 1 or args.fraction < 0 or args.fraction >= args.data_width:
        print("Error: compile-nir requires data-width > 1 and 0 <= fraction < data-width")
        return 1
    import os

    import nir as nir_lib

    from sc_neurocore.ir import SCNIR_HDL_HANDOFF_MANIFEST_VERSION, write_scnir
    from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork

    ext = os.path.splitext(args.model)[1].lower()
    if ext not in (".nir", ".onnx"):
        print(f"Error: compile-nir supports .nir and .onnx files, got '{ext}'")
        return 1

    data_width = int(args.data_width)
    fraction = int(args.fraction)

    print(f"[1/4] Loading model: {args.model}")
    # ``nir.read`` is the canonical importer for both NIR and ONNX inputs.
    graph = nir_lib.read(args.model)
    network = from_nir(graph, dt=args.dt)

    print(f"  Loaded {len(network.topo_order)} nodes")

    print("[2/4] Building NeuronGraph...")
    neuron_graph = from_scnetwork(network, dt=args.dt)
    print(f"  {neuron_graph.total_neurons} neurons, {neuron_graph.total_synapses} synapses")
    print(f"  Types: {', '.join(sorted(neuron_graph.neuron_types))}")

    print(f"[3/4] Compiling to Verilog (Q{data_width - fraction}.{fraction})...")
    interconnect = None if args.interconnect == "auto" else args.interconnect
    result = compile_network_to_fpga(
        neuron_graph,
        module_name=args.module_name,
        data_width=data_width,
        fraction=fraction,
        bitstream_length=args.T,
        source_kind=args.source_kind,
        base_seed=args.base_seed,
        target=args.target,
        interconnect=interconnect,
    )
    print(f"  Interconnect: {result.interconnect}")
    print(f"  Neuron modules: {len(result.neuron_modules)}")
    print(f"  SC-NIR source modules: {len(result.scnir_source_modules)}")
    folded_area = None
    if result.folded_metrics is not None:
        fm = result.folded_metrics
        print(
            f"  Folded datapath: {fm.populations} population(s), {fm.pe_instances} PE + "
            f"{fm.shared_multipliers} shared multiplier(s) + {fm.state_ram_bits}-bit state BRAM, "
            f"{fm.cycles_per_tick} cycles/tick "
            f"(collapses {fm.direct_neuron_instances} direct neuron instances)"
        )
        # Map the folded resource counts onto the Yosys-calibrated per-block costs to
        # report a pre-synthesis area/latency/power estimate (skips the non-FPGA 'web'
        # target, which has no resource model).
        from sc_neurocore.energy import estimate_folded_area
        from sc_neurocore.energy.fpga_models import TARGETS

        if args.target in TARGETS:
            folded_area = estimate_folded_area(fm, target=args.target, data_width=data_width)
            print(
                f"  Folded area (~est. {args.target}): {folded_area.total_luts} LUTs, "
                f"{folded_area.dsps} DSP, {folded_area.total_bram_kb:.2f} KB BRAM, "
                f"{folded_area.dynamic_power_mw:.2f} mW @ {folded_area.clock_freq_mhz:.0f} MHz "
                f"({folded_area.lut_utilisation_pct:.1f}% LUTs, "
                f"fits={'yes' if folded_area.fits_on_target else 'no'})"
            )

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

    if result.folded_metrics is not None:
        folded_path = os.path.join(out_dir, "folded_metrics.json")
        folded_payload: dict[str, Any] = dict(result.folded_metrics.as_dict())
        if folded_area is not None:
            folded_payload["area_estimate"] = folded_area.as_dict()
        with open(folded_path, "w", encoding="utf-8") as f:
            json.dump(folded_payload, f, indent=2, sort_keys=True)
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


def run_compile(args: argparse.Namespace) -> int:
    """Compile an ODE equation to Verilog RTL and optional synthesis.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``compile`` arguments.

    Returns
    -------
    int
        Zero after artefact emission; invalid equations propagate their typed error.

    Notes
    -----
    The command supports three compilation modes via CLI flags:

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
    if not args.model:
        print(
            "Error: compile requires an ODE string. Usage:\n"
            '  sc-neurocore compile "dv/dt = -(v-E_L)/tau_m + I/C" \\\n'
            '    --threshold "v > -50" --reset "v = -65" \\\n'
            '    --params "E_L=-65,tau_m=10,C=1" --init "v=-65" \\\n'
            "    --target ice40 --testbench --synthesize"
        )
        return 1
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

    # Optional synthesisable HLS C++ from the same ODE (Vitis/Catapult ap_fixed).
    if args.emit_hls:
        from sc_neurocore.compiler.intelligence.hls_export import generate_hls_cpp

        hls_src = generate_hls_cpp(
            args.module_name,
            neuron.equations,
            hls_tool=args.hls_tool,
            dt=args.dt,
            threshold=args.hls_threshold,
        )
        hls_path = os.path.join(out_dir, f"{args.module_name}.hls.cpp")
        with open(hls_path, "w") as f:
            f.write(hls_src)
        print(f"      HLS C++ written: {hls_path}")

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
        cfg = TARGET_CONFIGS.get(args.target)
        if cfg and cfg["tool"] == "yosys":
            synth_ok = run_auto_synthesis(out_dir, args.target, args.module_name, cfg)
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
    if args.emit_hls:
        print(f"  {args.module_name}.hls.cpp — synthesisable Vitis/Catapult HLS C++")
    if args.testbench:
        print(f"  tb_{args.module_name}.v — simulation testbench")
        print(f"  Run: iverilog -o sim {args.module_name}.v tb_{args.module_name}.v && vvp sim")
    return 0
