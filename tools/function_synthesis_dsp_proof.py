# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete primitive input and output correspondence

"""Prove all connected DSP inputs and original outputs under universal primitive values."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from tools.function_synthesis_aiger import AigerInterface
from tools.function_synthesis_dsp_ports import align_dsp_ports
from tools.function_synthesis_gia import prove_gia_equivalence
from tools.function_synthesis_netlist import Bit, Netlist, object_mapping, parameter_integer
from tools.function_synthesis_polynomial import ExactWordPolynomials
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import WordGraph

Label: TypeAlias = tuple[str, str, int]
WORD_EXPORT_ABC = "strash; map\n"
WORD_EXPORT_FLOW = (
    "read_json word-cut.json\nhierarchy -check -top function_unit\n"
    "proc\nmemory_map\nopt\nwreduce\nopt\ntechmap\nopt_clean\n"
    "abc -g AND -script word-abc.script\nopt_clean\ncheck -assert\n"
    "write_aiger -symbols word.aig\n"
)


def _write(path: Path, payload: object) -> None:
    """Retain a new exact private proof artifact without replacing prior results."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _output_groups(
    module: Netlist,
    labels: tuple[Label, ...],
    primary: Mapping[int, str],
) -> tuple[dict[Label, tuple[Label, ...]], dict[Label, tuple[str, str]], dict[Label, int]]:
    """Require actual primitive output alias partitions, constants and original-input aliases."""
    by_bit: dict[int, list[Label]] = {}
    fixed: dict[Label, tuple[str, str]] = {}
    for label in labels:
        name, port, index = label
        bit = module.cells[name].connections[port][index]
        if isinstance(bit, str):
            fixed[label] = ("constant", bit)
        elif bit in primary:
            fixed[label] = ("primary", primary[bit])
        else:
            by_bit.setdefault(bit, []).append(label)
    groups = {min(values): tuple(sorted(values)) for values in by_bit.values()}
    actual = {min(values): bit for bit, values in by_bit.items()}
    return groups, fixed, actual


def _primitive_order(
    module: Netlist, names: tuple[str, ...], primary: Mapping[int, str]
) -> dict[str, object]:
    """Require acyclic primitive dependencies and prove unmatched primitives unobserved."""
    drivers: dict[int, str] = {}
    primitive_owner: dict[int, str] = {}
    for name, cell in module.cells.items():
        for port, direction in cell.directions.items():
            if direction != "output":
                continue
            for bit in cell.connections[port]:
                if isinstance(bit, str) or bit in primary:
                    continue
                if bit in drivers and drivers[bit] != name:
                    raise ValueError("A residual proof contains multiple actual bit drivers")
                drivers[bit] = name
                if cell.kind == "DSP48E1":
                    primitive_owner[bit] = name
    memo: dict[int, frozenset[str]] = {}

    def roots(bit: Bit, active: frozenset[str]) -> frozenset[str]:
        """Trace actual functional data edges to independent primary or primitive outputs."""
        if bit in {"0", "1"} or bit in primary:
            return frozenset()
        if isinstance(bit, str):
            raise ValueError("A residual function depends on an unknown actual data bit")
        if bit in primitive_owner:
            return frozenset({primitive_owner[bit]})
        if bit in memo:
            return memo[bit]
        if bit not in drivers or drivers[bit] in active:
            raise ValueError("A residual function has an undriven edge or combinational cycle")
        name = drivers[bit]
        cell = module.cells[name]
        found: set[str] = set()
        for port, direction in cell.directions.items():
            if direction != "input":
                continue
            if cell.kind == "$memrd_v2" and port == "CLK":
                if parameter_integer(cell.parameters, "CLK_ENABLE"):
                    raise ValueError("A residual function contains a stateful memory read")
                continue
            for source in cell.connections[port]:
                found.update(roots(source, active | {name}))
        memo[bit] = frozenset(found)
        return memo[bit]

    dependencies: dict[str, set[str]] = {}
    for name in names:
        needed: set[str] = set()
        cell = module.cells[name]
        for port, direction in cell.directions.items():
            if direction == "input":
                for bit in cell.connections[port]:
                    needed.update(roots(bit, frozenset()))
        dependencies[name] = needed
    observable: set[str] = set()
    for bit in module.ports["result"].bits:
        observable.update(roots(bit, frozenset()))
    matched = set(names)
    if not observable <= matched or any(not needed <= matched for needed in dependencies.values()):
        raise ValueError("An unmatched DSP affects an original result or matched primitive input")
    active: set[str] = set()
    done: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        """Establish the actual induction order for all matched combinational primitives."""
        if name in done:
            return
        if name in active:
            raise ValueError("The actual primitive dependency graph is cyclic")
        active.add(name)
        for previous in sorted(dependencies[name]):
            visit(previous)
        active.remove(name)
        done.add(name)
        order.append(name)

    for name in names:
        visit(name)
    return {
        "primitive_dependencies": {
            name: sorted(previous) for name, previous in dependencies.items()
        },
        "primitive_order": order,
        "original_result_direct_primitive_dependencies": sorted(observable),
        "unobserved_unmatched_primitives": sorted(
            name
            for name, cell in module.cells.items()
            if cell.kind == "DSP48E1" and name not in matched
        ),
    }


@dataclass(frozen=True, slots=True)
class PrimitiveResidualProof:
    """Retain the actual primitive comparison domain and its acyclic composition.

    Parameters
    ----------
    primitive_count : int
        DSP count in the unchanged physical candidate, never a proof-clone count.
    free_output_values : int
        Universally independent actual primitive-output alias classes.
    primitive_input_bits : int
        Every connected input bit of every matched primitive, including controls.
    original_input_bits, original_output_bits : int
        Complete original function geometry, including all independent arguments.
    word_order, physical_order : dict of str to object
        Actual dependency edges, induction orders and proved-unobserved extra cells.

    """

    primitive_count: int
    free_output_values: int
    primitive_input_bits: int
    original_input_bits: int
    original_output_bits: int
    word_order: dict[str, object]
    physical_order: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Describe a complete native component proof without changing physical area."""
        return {
            "native_equivalence_engine": "gia-complete-all-output-or",
            "physical_dsp_count": self.primitive_count,
            "universal_primitive_values": self.free_output_values,
            "compared_primitive_input_bits": self.primitive_input_bits,
            "original_input_bits": self.original_input_bits,
            "original_output_bits": self.original_output_bits,
            "word_dependencies": self.word_order,
            "physical_dependencies": self.physical_order,
        }


def _bit_names(name: str, width: int) -> frozenset[str]:
    """Name every actual scalar or vector primary bit as emitted by Yosys."""
    return (
        frozenset({name}) if width == 1 else frozenset(f"{name}[{index}]" for index in range(width))
    )


def _prepare_residual(
    word: Netlist,
    physical: Netlist,
    input_widths: Mapping[str, int],
    output_width: int,
    directory: Path,
) -> tuple[PrimitiveResidualProof, AigerInterface]:
    """Serialize complete checked graphs before starting any native export process."""
    word, alignment = align_dsp_ports(word, physical)
    WordGraph(word, ExactWordPolynomials(), input_widths, output_width)
    WordGraph(physical, ExactWordPolynomials(), input_widths, output_width)
    source_primitives = {name: cell for name, cell in word.cells.items() if cell.kind == "DSP48E1"}
    target_primitives = {
        name: cell for name, cell in physical.cells.items() if cell.kind == "DSP48E1"
    }
    names = tuple(sorted(target_primitives))
    if not names:
        raise ValueError("The physical DSP instances do not match the actual word checkpoint")
    labels = tuple(
        (name, port, index)
        for name in names
        for port, direction in sorted(source_primitives[name].directions.items())
        if direction == "output"
        for index in range(len(source_primitives[name].connections[port]))
    )
    primary_word = word.primary_inputs(input_widths, output_width)
    primary_physical = physical.primary_inputs(input_widths, output_width)
    groups_a, fixed_a, bits_a = _output_groups(word, labels, primary_word)
    groups_b, fixed_b, bits_b = _output_groups(physical, labels, primary_physical)
    if groups_a != groups_b or fixed_a != fixed_b:
        raise ValueError("Actual primitive output aliases, constants or primary identities differ")
    word_order = _primitive_order(word, names, primary_word)
    physical_order = _primitive_order(physical, names, primary_physical)
    comparisons = [
        (name, port, f"dsp_input_{index:02d}_{port}")
        for index, name in enumerate(names)
        for port, direction in sorted(source_primitives[name].directions.items())
        if direction == "input"
    ]
    input_names = frozenset(
        name for port, width in input_widths.items() for name in _bit_names(port, width)
    )
    input_names |= frozenset(f"dsp_value_{index:04d}" for index in range(len(groups_a)))
    output_names = _bit_names("result", output_width)
    for name, port, label in comparisons:
        output_names |= _bit_names(label, len(source_primitives[name].connections[port]))
    interface = AigerInterface(input_names, output_names)
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "native-port-alignment.json", alignment)
    _write(directory / "aligned-word.json", {"modules": {"function_unit": word.clone_module()}})
    for role, module, bits in [("word", word, bits_a), ("physical", physical, bits_b)]:
        clone = module.clone_module()
        ports = object_mapping(clone["ports"])
        for index, group_label in enumerate(sorted(groups_a)):
            ports[f"dsp_value_{index:04d}"] = {"direction": "input", "bits": [bits[group_label]]}
        for name, port, label in comparisons:
            ports[label] = {
                "direction": "output",
                "bits": list(module.cells[name].connections[port]),
            }
        cells = object_mapping(clone["cells"])
        for name, cell in module.cells.items():
            if cell.kind == "DSP48E1":
                del cells[name]
        clone["ports"], clone["cells"] = ports, cells
        _write(directory / (role + "-cut.json"), {"modules": {"function_unit": clone}})
        flow = (
            "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
            f"read_json {role}-cut.json\n"
            "hierarchy -check -top function_unit\n"
            "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v\n"
            "proc\nmemory_map\nopt\nwreduce\nopt\ntechmap\nopt_clean\naigmap\nopt_clean\ncheck -assert\n"
            f"write_aiger -symbols {role}.aig\n"
        )
        if role == "word":
            if any(
                cell.kind != "DSP48E1" and not cell.kind.startswith("$")
                for cell in module.cells.values()
            ):
                raise ValueError("The word residual contains a non-generic actual cell")
            with (directory / "word-abc.script").open("x", encoding="utf-8") as stream:
                stream.write(WORD_EXPORT_ABC)
            flow = WORD_EXPORT_FLOW
        with (directory / (role + ".ys")).open("x", encoding="utf-8") as stream:
            stream.write(flow)
    proof = PrimitiveResidualProof(
        len(names),
        len(groups_a),
        sum(len(source_primitives[name].connections[port]) for name, port, _ in comparisons),
        sum(input_widths.values()),
        output_width,
        word_order,
        physical_order,
    )
    return proof, interface


def prove_primitive_residual(
    word: Netlist | Path,
    physical: Netlist | Path,
    input_widths: Mapping[str, int],
    output_width: int,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
) -> PrimitiveResidualProof:
    """Prove every matched primitive input function and the complete original result.

    Parameters
    ----------
    word, physical : Netlist or pathlib.Path
        Actual post-DSP checkpoint and unchanged physical mapped function module.
        Paths allow parsed graphs to be released before native children start.
    input_widths : mapping of str to int
        All independently driven original argument ports and complete widths.
    output_width : int
        Complete original result width.
    tools : SynthesisToolchain
        Existing inspected native executables and library contents.
    directory : pathlib.Path
        New private directory for proof clones, scripts, logs and complete exports.
    timeout : float
        Per-command native deadline, at most ninety seconds. CPU and memory
        containment are supplied by the caller, including this function's native children.

    Returns
    -------
    PrimitiveResidualProof
        Complete checked comparison and actual acyclic composition, only after
        native equality of every primitive input and original result bit.

    Raises
    ------
    OSError, ValueError
        For changed parameters, missing edges, aliases, state, observable unmatched
        primitives, unknown functional data or any unproved complete native comparison.
    SynthesisToolError
        If a native command fails or reaches its unchanged deadline.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("A residual proof requires a finite positive native deadline")
    timeout = min(timeout, 90.0)
    proof, interface = _prepare_residual(
        Netlist.read(word) if isinstance(word, Path) else word,
        Netlist.read(physical) if isinstance(physical, Path) else physical,
        input_widths,
        output_width,
        directory,
    )
    gc.collect()
    for role in ("word", "physical"):
        log = directory / (role + "-synthesis.log")
        tools.run([str(tools.yosys), "-Q", "-T", role + ".ys"], directory, log, timeout=timeout)
        if role == "physical":
            tools.check_library_log(log)
    prove_gia_equivalence(
        directory / "word.aig",
        directory / "physical.aig",
        interface,
        tools,
        directory,
        timeout=timeout,
    )
    _write(directory / "primitive-residual.json", proof.to_dict())
    return proof
