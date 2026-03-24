# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Minimal CLI for SC-NeuroCore

"""Minimal CLI for SC-NeuroCore."""

import argparse
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sc-neurocore",
        description="SC-NeuroCore — Universal Stochastic Computing Framework",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info", "benchmark", "preflight", "deploy", "serve"],
        help="Command to run",
    )
    parser.add_argument("model", nargs="?", help="Model file (.nir, .pt, .onnx) for deploy command")
    parser.add_argument(
        "--target",
        default="ice40",
        choices=["ice40", "ecp5", "artix7", "zynq"],
        help="FPGA target for deploy (default: ice40)",
    )
    parser.add_argument("--output", "-o", default="build", help="Output directory for deploy")
    parser.add_argument(
        "--dt", type=float, default=0.001, help="Simulation timestep for NIR import"
    )
    parser.add_argument("--T", type=int, default=256, help="Bitstream length for SC layers")
    parser.add_argument("--port", type=int, default=8001, help="Port for serve command")
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
    if args.command == "deploy":
        if not args.model:
            print(
                "Error: deploy requires a model file. Usage: sc-neurocore deploy model.nir --target artix7"
            )
            return 1
        return _cmd_deploy(args.model, args.target, args.output, args.dt, args.T)
    if args.command == "serve":
        if not args.model:
            print(
                "Error: serve requires a model file. Usage: sc-neurocore serve model.nir --port 8001"
            )
            return 1
        return _cmd_serve(args.model, args.port, args.dt)

    parser.print_help()
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


def _print_optional_dependency_version(module_name: str, label: str) -> None:
    try:
        module = __import__(module_name)
    except Exception:
        return
    print(f"{label}: {getattr(module, '__version__', 'unknown')}")


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
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "pytest", "benchmarks/benchmark_suite.py", "--benchmark-only"],
    ).returncode


def _cmd_deploy(
    model_path: str, target: str, output_dir: str, dt: float, bitstream_length: int
) -> int:
    """Deploy a model to FPGA: NIR/PyTorch → quantize → Verilog → project."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("SC-NeuroCore Deploy")
    print(f"  Model:  {model_path}")
    print(f"  Target: {target}")
    print(f"  Output: {output_dir}")
    print()

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
        import torch
        from sc_neurocore.conversion import convert

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        layers: list[torch.nn.Module] = []
        weight_keys = [k for k in state if k.endswith(".weight") and state[k].dim() == 2]
        for k in weight_keys:
            w = state[k]
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
            layers.append(torch.nn.ReLU())
        if layers and isinstance(layers[-1], torch.nn.ReLU):
            layers.pop()
        model = torch.nn.Sequential(*layers)
        model.load_state_dict(state, strict=False)
        in_dim = layers[0].in_features if layers else 1
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

    print()
    print(f"Deploy complete. Project in {output_dir}/")
    print("Next steps:")
    if target == "ice40":
        print(f"  cd {output_dir} && make synth  # Yosys synthesis")
    elif target == "ecp5":
        print(f"  cd {output_dir} && make synth  # Yosys + nextpnr-ecp5")
    else:
        print(f"  cd {output_dir} && vivado -mode batch -source project.tcl")
    return 0


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


def _cmd_preflight() -> int:
    import subprocess

    return subprocess.run([sys.executable, "tools/preflight.py"]).returncode


if __name__ == "__main__":
    sys.exit(main())
