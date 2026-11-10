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
from collections.abc import Sequence
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
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Samples .npy in the checkpoint's input format, values in [0, 1]: calibrates "
            "ReLU thresholds and measures the target fixed-point fit"
        ),
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
        calibration_path=args.calibration,
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
    calibration_path: str | None = None,
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
    calibration_path : str | None
        Samples for ReLU calibration and the target fixed-point report.

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
        print("[2/5] The NIR graph is validated only; no RTL is generated for it")
    elif ext in (".pt", ".pth"):
        print("[1/5] Loading the trusted PyTorch checkpoint and converting its network...")
        if not checkpoint_sha256:
            print(
                "Error: deploy requires --checkpoint-sha256 for .pt/.pth inputs "
                "(fail-closed trusted checkpoint loading)."
            )
            return 1
        if not _is_valid_sha256_digest(checkpoint_sha256):
            print("Error: --checkpoint-sha256 must be exactly 64 hexadecimal characters.")
            return 1
        from sc_neurocore.security.checkpoint_loading import (
            CheckpointTrustError,
            safe_load_checkpoint,
        )

        try:
            state = safe_load_checkpoint(
                model_path,
                trusted_sha256={model_path: checkpoint_sha256},
                map_location="cpu",
            )
        except CheckpointTrustError as exc:
            print(f"Error: {exc}")
            return 1
        converted = _convert_checkpoint(
            state, output_dir, target, bitstream_length, checkpoint_sha256, calibration_path
        )
        if converted is None:
            return 1
        deployment_layer_sizes = converted
        network = None
    else:
        print(f"Error: unsupported file format '{ext}'. Supported: .nir, .pt")
        return 1

    # Step 3: Generate Verilog
    print("[3/5] Generating the generic LIF RTL template (it carries no trained weights)...")
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
    _generate_project(output_dir, target, "sc_deploy_lif", converted=ext in (".pt", ".pth"))
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
    import shutil

    cfg = TARGET_CONFIGS[target]
    yosys_installed = cfg["tool"] == "yosys" and shutil.which("yosys") is not None
    if yosys_installed:
        synth_ok = run_auto_synthesis(output_dir, target, "sc_deploy_lif", cfg)
    else:
        synth_ok = False

    print()
    print(f"Deploy complete. Project in {output_dir}/")
    if synth_ok:
        print("Synthesis succeeded. Results in output directory.")
    elif yosys_installed:
        print("Synthesis flow failed (errors above). The generated project is kept.")
        return 1
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
    *,
    sources: Sequence[str] | None = None,
) -> bool:
    """Run the open-source synthesis flow when its tools are installed.

    Parameters
    ----------
    output_dir : str
        Directory holding the Verilog sources; outputs are written there.
    target : str
        Target identifier used in status output.
    top_module : str
        Top-module name.
    cfg : dict[str, str]
        Device family, part, package, and tool configuration.
    sources : Sequence[str] or None, optional
        Verilog files to read, relative to ``output_dir``. ``None`` reads the
        deployment layout that ``deploy`` writes: every ``hdl/*.v`` plus
        ``<top_module>.sv``. ``compile`` passes the one ``<module>.v`` it writes.

    Returns
    -------
    bool
        ``True`` when Yosys and every later step that ran (nextpnr place and
        route, bitstream packing) succeeded. A later step whose tool is not
        installed is skipped with a message. ``False`` when Yosys is not
        installed, or when any step that ran failed or exceeded its time limit.
    """
    import os
    import shutil

    yosys = shutil.which("yosys")
    if not yosys:
        return False

    print()
    print("[6/6] Running Yosys synthesis...")
    if sources is None:
        sources = [
            os.path.join("hdl", f)
            for f in os.listdir(os.path.join(output_dir, "hdl"))
            if f.endswith(".v")
        ] + [f"{top_module}.sv"]
    verilog_files = " ".join(sources)
    synth_cmd = f"synth_{cfg['family']}"
    yosys_script = (
        f"read_verilog -sv {verilog_files}; "
        f"{synth_cmd} -top {top_module}; "
        # Yosys 0.70 emits portless $scopeinfo debug cells when it flattens a
        # hierarchy; nextpnr 0.6 cannot place them.
        "delete t:$scopeinfo; "
        f"write_json {top_module}.json; stat"
    )
    result = _run_flow_step(
        [yosys, "-p", yosys_script], output_dir, SYNTHESIS_TIMEOUT_S, "Yosys synthesis"
    )
    if result is None:
        return False
    if result.returncode != 0:
        print("  Yosys synthesis failed:")
        _print_stderr_tail(result.stderr)
        return False
    for line in result.stdout.splitlines():
        if any(k in line for k in ("Number of cells", "Number of wires", "LUT", "SB_")):
            print(f"  {line.strip()}")
    print(f"  Synthesis JSON: {os.path.join(output_dir, top_module + '.json')}")

    # Place and route only when nextpnr is installed; a missing tool skips the step.
    pnr_name = f"nextpnr-{cfg['family']}"
    pnr_tool = shutil.which(pnr_name)
    if not pnr_tool:
        print(f"  Place-and-route skipped ({pnr_name} not found).")
        return True
    print("  Running nextpnr place-and-route...")
    # nextpnr-ice40 writes an IceStorm .asc that icepack packs; nextpnr-ecp5
    # takes no --asc and writes a Trellis text configuration for ecppack.
    placed, pack_tool, bitstream = _PLACED_OUTPUTS[cfg["family"]]
    placed_file = f"{top_module}.{placed[1]}"
    pnr_result = _run_flow_step(
        [
            pnr_tool,
            f"--{cfg['device']}",
            "--json",
            f"{top_module}.json",
            placed[0],
            placed_file,
            "--package",
            cfg["package"],
        ],
        output_dir,
        PLACE_AND_ROUTE_TIMEOUT_S,
        "nextpnr place-and-route",
    )
    if pnr_result is None:
        return False
    if pnr_result.returncode != 0:
        print("  PnR failed (nextpnr error). Synthesis JSON still available.")
        _print_stderr_tail(pnr_result.stderr)
        return False
    print(f"  PnR succeeded: {placed_file}")

    # Pack a bitstream only when the family's packer is installed.
    pack_bin = shutil.which(pack_tool)
    if not pack_bin:
        print(f"  Bitstream skipped ({pack_tool} not found).")
        return True
    bitstream_file = f"{top_module}.{bitstream}"
    pack_result = _run_flow_step(
        [pack_bin, placed_file, bitstream_file],
        output_dir,
        PACK_TIMEOUT_S,
        f"{pack_tool} bitstream packing",
    )
    if pack_result is None:
        return False
    bin_path = os.path.join(output_dir, bitstream_file)
    if pack_result.returncode != 0 or not os.path.exists(bin_path):
        print(f"  Bitstream packing failed ({pack_tool} wrote no {bitstream_file}).")
        _print_stderr_tail(pack_result.stderr)
        return False
    size_kb = os.path.getsize(bin_path) / 1024
    print(f"  Bitstream: {bin_path} ({size_kb:.1f} KB)")
    return True


#: Wall-clock limits, in seconds, for the open-source flow steps. They stay fixed
#: here until host-derived limits replace them (TODO SCN-HOST-DERIVED-TEST-DEADLINES).
SYNTHESIS_TIMEOUT_S = 300
PLACE_AND_ROUTE_TIMEOUT_S = 300
PACK_TIMEOUT_S = 60


def _run_flow_step(
    command: list[str], cwd: str, timeout_s: int, label: str
) -> subprocess.CompletedProcess[str] | None:
    """Run one synthesis-flow tool, reporting a timeout instead of raising it.

    Parameters
    ----------
    command : list[str]
        Tool executable and arguments.
    cwd : str
        Working directory of the tool.
    timeout_s : int
        Wall-clock limit in seconds.
    label : str
        Step name used in the timeout message.

    Returns
    -------
    subprocess.CompletedProcess[str] or None
        The finished process, or ``None`` after printing that the step timed out.
    """
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  {label} timed out after {timeout_s} s.")
        return None


def _print_stderr_tail(stderr: str | None) -> None:
    """Print the last five lines of a tool's standard error."""
    for line in (stderr or "").splitlines()[-5:]:
        print(f"    {line}")


#: Per FPGA family: nextpnr's placed-design option and file suffix, the packer,
#: and the bitstream suffix the packer writes.
_PLACED_OUTPUTS: dict[str, tuple[tuple[str, str], str, str]] = {
    "ice40": (("--asc", "asc"), "icepack", "bin"),
    "ecp5": (("--textcfg", "config"), "ecppack", "bit"),
}

TARGET_CONFIGS: dict[str, dict[str, str]] = {
    "ice40": {"family": "ice40", "device": "hx8k", "package": "ct256", "tool": "yosys"},
    "ecp5": {"family": "ecp5", "device": "85k", "package": "CABGA381", "tool": "yosys"},
    "artix7": {"family": "xc7a", "device": "xc7a100t", "package": "csg324", "tool": "vivado"},
    "zynq": {"family": "xc7z", "device": "xc7z020", "package": "clg400", "tool": "vivado"},
}


def _generate_project(
    output_dir: str, target: str, top_module: str, *, converted: bool = False
) -> None:
    """Write the target-specific build script and deployment README."""
    import os

    cfg = TARGET_CONFIGS[target]

    if cfg["tool"] == "yosys":
        (placed_option, placed_suffix), packer, bitstream = _PLACED_OUTPUTS[cfg["family"]]
        makefile = f"""# SC-NeuroCore Deploy — {target} target
TOP = {top_module}
DEVICE = {cfg["device"]}

VERILOG_FILES = $(wildcard hdl/*.v) {top_module}.sv

.PHONY: synth pnr bitstream clean

synth:
\tyosys -p "read_verilog -sv $(VERILOG_FILES); synth_{cfg["family"]} -top $(TOP); delete t:\\$$scopeinfo; write_json $(TOP).json; stat"

pnr: synth
\tnextpnr-{cfg["family"]} --{cfg["device"]} --json $(TOP).json {placed_option} $(TOP).{placed_suffix} --package {cfg["package"]}

bitstream: pnr
\t{packer} $(TOP).{placed_suffix} $(TOP).{bitstream}

clean:
\trm -f *.json *.{placed_suffix} *.{bitstream}
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
- `{top_module}.sv` — Generic LIF neuron template (Q8.8 fixed-point); it does
  not carry the deployed model's weights
- `hdl/` — SC-NeuroCore Verilog library (encoders, synapses, layers)
- `{"Makefile" if cfg["tool"] == "yosys" else "project.tcl"}` — Build script
{_CONVERTED_FILES if converted else ""}
## Build
{"make synth" if cfg["tool"] == "yosys" else "vivado -mode batch -source project.tcl"}
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme)


_CONVERTED_FILES = """- `converted_network.npz` / `converted_network.json` — The checkpoint's converted
  dense IF network and its manifest; no RTL is generated for it
- `target_report.json` — Its fixed-point calibration for this target, when
  calibration samples were given
"""


def _convert_checkpoint(
    state: object,
    output_dir: str,
    target: str,
    steps: int,
    checkpoint_sha256: str,
    calibration_path: str | None,
) -> list[tuple[int, int]] | None:
    """Convert a loaded checkpoint, export the network and calibrate it for the target.

    Returns
    -------
    list of tuple of int or None
        Dense layer sizes, or ``None`` after printing why the checkpoint was refused.
    """
    import json
    import os

    import numpy as np

    from sc_neurocore.conversion.checkpoint_network import network_from_checkpoint
    from sc_neurocore.conversion.converted_io import save_converted_network

    samples = None
    try:
        if calibration_path is not None:
            samples = np.load(calibration_path, allow_pickle=False).astype(np.float64)
        loaded = network_from_checkpoint(
            state,
            steps=steps,
            calibration=None if samples is None else samples.reshape(len(samples), -1),
            max_dense_params=_MAX_DEPLOY_DENSE_PARAMS,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return None
    snn = loaded.snn
    digest = save_converted_network(snn, os.path.join(output_dir, "converted_network.npz"))
    manifest = {
        "schema_version": "sc-neurocore.deploy-converted-network.v1",
        "source_checkpoint_sha256": checkpoint_sha256.lower(),
        "source": loaded.source,
        "calibration": loaded.calibration,
        "layer_sizes": loaded.layer_sizes,
        "T": snn.T,
        "output_mode": snn.output_mode,
        "converted_sha256": digest,
        "rtl": "the generated RTL is a generic LIF template and does not carry this network",
    }
    with open(os.path.join(output_dir, "converted_network.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  Converted {snn.n_layers}-layer network, T={snn.T}, calibration: {loaded.calibration}")
    print(f"  Network -> converted_network.npz (sha256 {digest[:12]})")
    print("[2/5] Calibrating the converted network for the target format...")
    if samples is None:
        print("  Skipped: pass --calibration samples.npy to measure the target fit")
        return loaded.layer_sizes
    from sc_neurocore.compiler.platforms import get_profile
    from sc_neurocore.conversion.target_report import calibrate_for_target

    try:
        profile = get_profile(target)
    except KeyError:
        print(f"  Skipped: no fixed-point profile is registered for '{target}'")
        return loaded.layer_sizes
    try:
        report = calibrate_for_target(snn, profile, samples.reshape(len(samples), -1))
    except ValueError as exc:
        print(f"Error: {exc}")
        return None
    with open(os.path.join(output_dir, "target_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.to_public_dict(), f, indent=2, sort_keys=True, allow_nan=False)
    verdict = "compatible" if report.compatible else "; ".join(report.refusals)
    print(f"  {profile.name} {profile.q_format_label}: {verdict} -> target_report.json")
    return loaded.layer_sizes


def _find_hdl_source() -> Path | None:
    """Find the repository HDL tree without depending on CLI package depth."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "hdl"
        if candidate.is_dir():
            return candidate
    return None
