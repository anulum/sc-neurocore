# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical mapping and independent complete AIG exports

"""Preserve physical mapping while exporting complete native models only when needed."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import digest
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain

SYNTHESIS_FLOW = """read_verilog -sv unit.v
hierarchy -check -top function_unit
proc
opt_clean
design -save prepared
synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf
write_json mapped.json
write_verilog -noattr mapped.v
tee -o stat.json stat -json -top function_unit
design -stash candidate
design -load prepared
memory_map
opt
wreduce
opt
techmap
opt_clean
aigmap
opt_clean
check -assert
write_aiger -symbols reference.aig
design -load candidate
hierarchy -check -top function_unit
techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v
proc
memory_map
opt
wreduce
opt
techmap
opt_clean
aigmap
opt_clean
check -assert
write_aiger -symbols mapped.aig
"""


def module_json_flow(filename: str, module: str = "function_unit") -> str:
    """Export a complete live module and its direct cell models without library clutter.

    Parameters
    ----------
    filename : str
        Local JSON filename containing lowercase letters, digits, underscores
        or hyphens, starting with a letter.
    module : str, optional
        Complete top module with a plain Verilog identifier.

    Returns
    -------
    str
        Native Yosys commands that preserve the complete module and each
        directly referenced module, then restore the entire original design.
        No cell, port, net, parameter or signed annotation is rewritten.

    Raises
    ------
    ValueError
        If a name cannot be represented as a safe local native command token.
    TypeError
        If either argument is not a string.

    Notes
    -----
    Copying the live design avoids the internal signed-net annotation loss
    observed in native JSON read/write round trips. Primitive libraries stay
    separately bound by the inspected toolchain's full file inventory. The
    explicit selection includes blackbox and whitebox cell models on both
    legacy and current Yosys releases.

    """
    if (
        re.fullmatch(r"[a-z][a-z0-9_-]*[.]json", filename) is None
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module) is None
    ):
        raise ValueError("A native module export requires a local JSON filename and identifier")
    return (
        "design -stash function_json_export\n"
        f"design -copy-from function_json_export ={module} %s\n"
        f"write_json {filename}\n"
        "design -load function_json_export\n"
        "design -delete function_json_export\n"
    )


MAPPING_FLOW = SYNTHESIS_FLOW.split("design -stash candidate\n", 1)[0].replace(
    "write_json mapped.json\n", module_json_flow("mapped.json")
)
_COMPLETE_BITS = "techmap\nopt_clean\naigmap\nopt_clean\ncheck -assert\n"
EXPORT_FLOW = (
    "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
    "memory_map\nopt\nwreduce\nopt\n"
    + _COMPLETE_BITS
    + "write_aiger -symbols reference.aig\ndesign -reset\n"
    "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
    "read_json physical-module.json\nhierarchy -check -top function_unit\n"
    "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v\n"
    "proc\nmemory_map\nopt\nwreduce\nopt\n" + _COMPLETE_BITS + "write_aiger -symbols mapped.aig\n"
)
EXPORT_ARTIFACTS = ("export.ys", "physical-module.json", "export.log", "export.command.json")


def _module_text(case: SynthesisCase, directory: Path) -> str:
    """Retain exactly the complete physical function without unrelated library declarations."""
    module = Netlist.read(directory / "mapped.json")
    module.primary_inputs(
        {
            f"argument{index}": case.data_width
            for index in range(case.input_bits // case.data_width)
        },
        case.data_width,
    )
    return (
        json.dumps({"modules": {"function_unit": module.clone_module()}}, sort_keys=True, indent=2)
        + "\n"
    )


def validate_complete_aiger_exports(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain
) -> None:
    """Validate retained native export inputs without reexecuting or claiming equivalence.

    Parameters
    ----------
    case : SynthesisCase
        Complete original function and primary geometry.
    directory : pathlib.Path
        Actual physical acquisition with independent native export artifacts.
    tools : SynthesisToolchain
        Inspected existing native tools and complete technology library.

    Raises
    ------
    OSError, ValueError
        For linked artifacts, changed canonical source, physical graph or export flow.

    """
    tools.verify()
    if directory.is_symlink() or any(
        (directory / name).is_symlink() for name in (*EXPORT_ARTIFACTS, "unit.v")
    ):
        raise ValueError("Complete AIG export artifacts must not be symbolic links")
    rtl = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    if (directory / "unit.v").read_text(encoding="utf-8") != rtl:
        raise ValueError("A complete native export used a different original function")
    if (directory / "export.ys").read_text(encoding="utf-8") != EXPORT_FLOW:
        raise ValueError("The complete native export flow changed")
    if (directory / "physical-module.json").read_text(encoding="utf-8") != _module_text(
        case, directory
    ):
        raise ValueError("The complete native export used a different physical function")
    tools.check_library_log(directory / "export.log")


def export_complete_aigers(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain, *, timeout: float
) -> None:
    """Export actual complete models independently of the unchanged physical mapping.

    Parameters
    ----------
    case : SynthesisCase
        Original function and complete input and output geometry.
    directory : pathlib.Path
        Completed physical acquisition; new export artifacts must not already exist.
    tools : SynthesisToolchain
        Existing inspected native executables and technology models.
    timeout : float
        Finite positive native deadline, capped at ninety seconds.

    Notes
    -----
    DSP-bearing functions use compositional proofs without this whole-DSP expansion.
    Actual complete native equality remains a separate checker responsibility.

    Raises
    ------
    OSError, ValueError
        For invalid budgets, existing artifacts or changed native export inputs.
    SynthesisToolError
        If the actual native export fails or reaches its unchanged command ceiling.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Complete native exports require a finite positive deadline")
    before = {name: digest(directory / name) for name in ["unit.v", "mapped.json"]}
    module_text = _module_text(case, directory)
    for name, content in [("physical-module.json", module_text), ("export.ys", EXPORT_FLOW)]:
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-s", "export.ys"],
        directory,
        directory / "export.log",
        timeout=min(timeout, 90.0),
    )
    if before != {name: digest(directory / name) for name in before}:
        raise ValueError("Original native export inputs changed during their acquisition")
    validate_complete_aiger_exports(case, directory, tools)
