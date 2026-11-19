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
from tools.function_synthesis_aiger import AigerInterface, digest
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

LARGE_LUT_EXPANSION = 1 << 19
_LUT_GATE_COST = {f"LUT{width}": (1 << width) - 1 for width in range(1, 7)}
_LUT_GATE_COST["LUT6_2"] = _LUT_GATE_COST["LUT6"] + _LUT_GATE_COST["LUT5"]
_TOP = "hierarchy -check -top function_unit\n"
_SELECT_TOP = "select -module function_unit\n"
_COMPACT_WIRES = (
    "select -clear\n"
    "tee -q attrmap -remove src function_unit\n"
    "rename -enumerate -pattern _checkpoint_% function_unit/w:* function_unit/c:*\n"
)


def lut_gate_expansion(module: Netlist) -> int:
    """Estimate the complete LUT decision graph size before sharing or simplification."""
    return sum(_LUT_GATE_COST.get(cell.kind, 0) for cell in module.cells.values())


def _read_rtlil(filename: str) -> str:
    """Resume a complete native stage and require the original top module."""
    return f"read_rtlil {filename}\n" + _TOP


def complete_checkpoint_blif_flows() -> tuple[dict[str, str], dict[str, str]]:
    """Describe bounded complete exports for large native function graphs.

    The mapped path simplifies word multiplexers with native don't-care
    preservation before expanding to gates. The final BLIF retains native
    aliases and is checked by a separate native reader before AIG conversion.
    """
    flows: dict[str, str] = {
        "reference-prepare": "read_verilog -sv unit.v\n"
        + _TOP
        + "proc\nmemory_map\nopt\nwreduce\nopt\nwrite_rtlil reference.prepared.il.gz\n",
        "mapped-prepare": (
            "read_verilog -lib -specify +/xilinx/cells_sim.v\nread_json physical-module.json\n"
            + _TOP
            + "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v\n"
            + _SELECT_TOP
            + "write_rtlil -selected mapped.primitives.il\n"
        ),
    }
    checkpoints = {
        "reference-prepare": "reference.prepared.il.gz",
        "mapped-prepare": "mapped.primitives.il",
    }
    for name, source, native_passes, target in (
        ("reference-map", "reference.prepared.il.gz", "techmap\n", "reference.gates.il"),
        (
            "reference-clean",
            "reference.gates.il",
            "opt_clean\n" + _COMPACT_WIRES,
            "reference.clean.il",
        ),
        (
            "mapped-lower",
            "mapped.primitives.il",
            "proc\nmemory_map\n" + _COMPACT_WIRES,
            "mapped.lowered.il",
        ),
        ("mapped-fold", "mapped.lowered.il", "opt_expr\nopt_clean\n", "mapped.folded.il"),
        ("mapped-width", "mapped.folded.il", "wreduce\n", "mapped.reduced.il"),
        (
            "mapped-word",
            "mapped.reduced.il",
            "pmuxtree\nopt_muxtree\nopt_expr -keepdc\nopt_clean\n",
            "mapped.word.il",
        ),
    ):
        flows[name] = (
            _read_rtlil(source)
            + _SELECT_TOP
            + native_passes
            + _SELECT_TOP
            + f"write_rtlil -selected {target}\n"
        )
        checkpoints[name] = target
    flows["reference-blif"] = (
        _read_rtlil("reference.clean.il") + "write_blif -top function_unit reference.blif\n"
    )
    flows["mapped-blif"] = (
        _read_rtlil("mapped.word.il")
        + _SELECT_TOP
        + "techmap\nwrite_blif -top function_unit mapped.blif\n"
    )
    for role in ("reference", "mapped"):
        flows[role + "-read-blif"] = f"read_blif {role}.blif\n" + _TOP + "check -assert\n"
    return flows, checkpoints


def validate_blif_output_drivers(path: Path, interface: AigerInterface, max_bytes: int) -> None:
    """Prove that no original output is fed by an undriven or undefined BLIF name.

    The native exporter writes a single combinational model of truth tables.
    Follow every declared input to each reachable table conservatively. ABC can
    assign zero to unused undriven aliases, which must stay outside this cone.
    """
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("The complete native BLIF requires a bounded regular artifact")
    drivers: dict[bytes, tuple[bytes, ...]] = {}
    declarations: dict[bytes, tuple[bytes, ...]] = {}
    with path.open("rb") as stream:
        for line in stream:
            statement = line.lstrip()
            if not statement.startswith(b"."):
                continue
            directive, *names = statement.split()
            if b".end" in declarations:
                raise ValueError("The complete native BLIF contains data after its model")
            if directive == b".names":
                if not names or names[-1] in drivers:
                    raise ValueError("The complete native BLIF has an invalid or repeated driver")
                drivers[names[-1]] = tuple(names[:-1])
            elif directive in {b".model", b".inputs", b".outputs", b".end"} and (
                directive not in declarations
            ):
                declarations[directive] = tuple(names)
            else:
                raise ValueError("The complete native BLIF uses an unsupported directive")
    primary = declarations.get(b".inputs", ())
    results = declarations.get(b".outputs", ())
    expected_inputs = {name.encode() for name in interface.inputs}
    expected_outputs = {name.encode() for name in interface.outputs}
    if (
        declarations.get(b".model") != (b"function_unit",)
        or declarations.get(b".end") != ()
        or len(primary) != len(expected_inputs)
        or set(primary) != expected_inputs
        or len(results) != len(expected_outputs)
        or set(results) != expected_outputs
        or not expected_inputs.isdisjoint(drivers)
    ):
        raise ValueError("The complete native BLIF changed its original model or ports")
    pending = list(results)
    visited = set(primary)
    while pending:
        wire = pending.pop()
        if wire in visited:
            continue
        upstream = drivers.get(wire)
        if upstream is None or wire.startswith(b"$undef"):
            raise ValueError("An original output depends on an undriven or undefined BLIF net")
        visited.add(wire)
        pending.extend(upstream)


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
