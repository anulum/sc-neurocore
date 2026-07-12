# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware deployment command

"""Deploy NIR or trusted PyTorch models into FPGA and browser artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_DEPLOY_DENSE_PARAMS = 20_000_000


def add_deploy_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the model deployment command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "deploy",
        help="Build an FPGA or browser deployment from a model",
        description="Convert a NIR graph or trusted dense PyTorch checkpoint into deployable artefacts.",
    )
    parser.add_argument("model", nargs="?", help="NIR graph or PyTorch checkpoint")
    parser.add_argument(
        "--target",
        default="ice40",
        choices=["ice40", "ecp5", "artix7", "zynq", "web"],
    )
    parser.add_argument("--output", "-o", default="build", help="Deployment output directory")
    parser.add_argument("--dt", type=float, default=1.0, help="Model timestep")
    parser.add_argument("--T", type=int, default=256, help="Stochastic bitstream length")
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help="Required SHA-256 digest for .pt/.pth checkpoint inputs",
    )
    parser.set_defaults(handler=run_deploy)


def run_deploy(args: argparse.Namespace) -> int:
    """Deploy one model through the selected target workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed deployment arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for an invalid or failed deployment.
    """
    if not args.model:
        print(
            "Error: deploy requires a model file. Usage: "
            "sc-neurocore deploy model.nir --target artix7"
        )
        return 1
    return _deploy_model(
        str(args.model),
        str(args.target),
        str(args.output),
        float(args.dt),
        int(args.T),
        checkpoint_sha256=args.checkpoint_sha256,
    )


def _is_valid_sha256_digest(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _deploy_model(
    model_path: str,
    target: str,
    output_dir: str,
    dt: float,
    bitstream_length: int,
    *,
    checkpoint_sha256: str | None = None,
) -> int:
    """Deploy a model to FPGA or browser artefacts.

    Parameters
    ----------
    model_path : str
        NIR graph or trusted PyTorch checkpoint path.
    target : str
        Hardware family or ``web`` deployment target.
    output_dir : str
        Destination directory for generated artefacts.
    dt : float
        Imported model timestep.
    bitstream_length : int
        Stochastic bitstream length used by the generated workload model.
    checkpoint_sha256 : str | None
        Required digest for PyTorch checkpoint inputs.

    Returns
    -------
    int
        Zero on success, otherwise one for a rejected or failed deployment.
    """
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
        from sc_neurocore.conversion.ann_to_snn import convert

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
        total_dense_params = sum(int(state[key].numel()) for key in weight_keys)
        if total_dense_params > _MAX_DEPLOY_DENSE_PARAMS:
            print(
                "Error: deploy checkpoint dense parameter count exceeds safety limit "
                f"({_MAX_DEPLOY_DENSE_PARAMS:,}): {total_dense_params:,}"
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
        # Every accepted dense weight appends exactly one trailing activation.
        layers.pop()
        model = torch.nn.Sequential(*layers)
        in_dim = linear_layers[0].in_features
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

    print("[4/5] Copying HDL modules...")
    hdl_src = _find_hdl_source()
    hdl_dst = os.path.join(output_dir, "hdl")
    if hdl_src is not None:
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
    cfg = TARGET_CONFIGS[target]
    if cfg["tool"] == "yosys":
        synth_ok = run_auto_synthesis(output_dir, target, "sc_deploy_lif", cfg)
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


def run_auto_synthesis(
    output_dir: str,
    target: str,
    top_module: str,
    cfg: dict[str, str],
) -> bool:
    """Run the open-source synthesis flow when its tools are installed.

    Parameters
    ----------
    output_dir : str
        Deployment directory containing the HDL tree.
    target : str
        Target identifier used in status output.
    top_module : str
        SystemVerilog top-module name.
    cfg : dict[str, str]
        Device family, part, package, and tool configuration.

    Returns
    -------
    bool
        ``True`` when Yosys succeeds, otherwise ``False``.
    """
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
    result = subprocess.run(
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
            pnr_result = subprocess.run(
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
                    subprocess.run(
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


TARGET_CONFIGS: dict[str, dict[str, str]] = {
    "ice40": {"family": "ice40", "device": "hx8k", "package": "ct256", "tool": "yosys"},
    "ecp5": {"family": "ecp5", "device": "85k", "package": "CABGA381", "tool": "yosys"},
    "artix7": {"family": "xc7a", "device": "xc7a100t", "package": "csg324", "tool": "vivado"},
    "zynq": {"family": "xc7z", "device": "xc7z020", "package": "clg400", "tool": "vivado"},
}


def _generate_project(output_dir: str, target: str, top_module: str) -> None:
    """Write the target-specific build script and deployment README."""
    import os

    cfg = TARGET_CONFIGS[target]

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


def _find_hdl_source() -> Path | None:
    """Find the repository HDL tree without depending on CLI package depth."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "hdl"
        if candidate.is_dir():
            return candidate
    return None
