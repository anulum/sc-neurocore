# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound compositional function proofs

"""Qualify actual DSP mappings without changing the original physical synthesis flow."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import (
    AigerInterface,
    digest,
)
from tools.function_synthesis_dsp_contracts import prove_dsp_contract, read_dsp_contract
from tools.function_synthesis_dsp_ports import read_dsp_port_alignment
from tools.function_synthesis_dsp_proof import (
    WORD_EXPORT_ABC,
    WORD_EXPORT_FLOW,
    prove_primitive_residual,
)
from tools.function_synthesis_exports import MAPPING_FLOW, SYNTHESIS_FLOW, module_json_flow
from tools.function_synthesis_json import native_netlist_artifacts
from tools.function_synthesis_netlist import Netlist, object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_source_proof import (
    prove_source_normalization,
    read_native_equivalence,
    read_source_normalization,
)
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import configured_dsp
from tools.function_synthesis_word_proof import prove_word_composition

_PROOF_DIRECTORY = "composition-proof"
_ROOT_INPUTS = (
    "unit.v",
    "flow.ys",
    "mapped.json",
    "mapped.v",
    "stat.json",
    "yosys.log",
    "yosys.command.json",
)
_CODE_FILES = tuple(
    "function_synthesis_" + name + ".py"
    for name in (
        "netlist",
        "polynomial",
        "word_graph",
        "word_proof",
        "structural_proof",
        "aiger",
        "dsp_proof",
        "dsp_ports",
        "dsp_contracts",
        "proof",
        "toolchain",
        "records",
        "yosys",
        "exports",
        "gia",
        "json",
        "source_proof",
    )
)
_PREFIX = "read_verilog -sv unit.v\nhierarchy -check -top function_unit\nproc\nopt_clean\n"
_FLOWS = {
    "pre-dsp.ys": _PREFIX
    + "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:map_dsp\n"
    + module_json_flow("pre-dsp.json"),
    "post-dsp.ys": _PREFIX
    + "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run begin:coarse\n"
    + module_json_flow("post-dsp.json"),
    "original-words.ys": _PREFIX
    + "memory_map\nopt\nwreduce\nopt\n"
    + module_json_flow("original-word.json")
    + "design -reset\nread_json pre-dsp.json\nhierarchy -check -top function_unit\nmemory_map\nopt\nwreduce\nopt\n"
    + module_json_flow("reference-word.json"),
}


def _encoded(value: object) -> str:
    """Serialize one exact evidence record in its retained canonical representation."""
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _write(path: Path, payload: str) -> None:
    """Create a new proof artifact while preserving every existing acquisition."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _object(path: Path) -> dict[str, object]:
    """Read one regular structured proof record without following an artifact link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("A compositional proof requires regular retained artifacts")
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return object_mapping(value)


def _source_bindings() -> dict[str, str]:
    """Bind every actual Python producer and arithmetic-checker implementation."""
    return {name: digest(Path(__file__).parent / name) for name in _CODE_FILES}


def _widths(case: SynthesisCase) -> dict[str, int]:
    """Preserve the complete independent original argument ports of the selected unit."""
    return {
        f"argument{index}": case.data_width for index in range(case.input_bits // case.data_width)
    }


def _names(ports: Mapping[str, int]) -> frozenset[str]:
    """Name every actual scalar or vector bit in the expected complete interface."""
    return frozenset(
        name if width == 1 else f"{name}[{index}]"
        for name, width in ports.items()
        for index in range(width)
    )


def _root_bindings(case: SynthesisCase, directory: Path) -> dict[str, str]:
    """Bind the original physical acquisition and exact canonical emitted function."""
    if directory.is_symlink() or any((directory / name).is_symlink() for name in _ROOT_INPUTS):
        raise ValueError("A compositional mapping acquisition must not contain linked artifacts")
    expected = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    if (directory / "unit.v").read_text(encoding="utf-8") != expected:
        raise ValueError("The compositional reference differs from the selected original function")
    if (directory / "flow.ys").read_text(encoding="utf-8") not in {MAPPING_FLOW, SYNTHESIS_FLOW}:
        raise ValueError("The physical synthesis differs from the unchanged mapping flow")
    return {
        name: digest(directory / name)
        for name in (*_ROOT_INPUTS, *native_netlist_artifacts(directory))
    }


def _artifacts(directory: Path) -> dict[str, str]:
    """Retain the complete regular proof closure without a self-referential record hash."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(
                "A compositional proof closure contains a linked or irregular artifact"
            )
        if path.is_file() and path != directory / "proof-record.json":
            result[path.relative_to(directory).as_posix()] = digest(path)
    return result


def _primitive_interface(
    case: SynthesisCase, post: Netlist, physical: Netlist, details: dict[str, object]
) -> AigerInterface:
    """Require all original bits, independent primitive values and every connected DSP input."""
    names = sorted(name for name, cell in physical.cells.items() if cell.kind == "DSP48E1")
    free = details.get("universal_primitive_values")
    primary = physical.primary_inputs(_widths(case), case.data_width)
    expected_free = len(
        {
            bit
            for name in names
            for port, direction in physical.cells[name].directions.items()
            if direction == "output"
            for bit in physical.cells[name].connections[port]
            if isinstance(bit, int) and bit not in primary
        }
    )
    if isinstance(free, bool) or not isinstance(free, int) or free != expected_free:
        raise ValueError("The primitive proof has invalid universally independent output geometry")
    inputs = _names(_widths(case)) | frozenset(f"dsp_value_{index:04d}" for index in range(free))
    outputs = _names({"result": case.data_width})
    count = 0
    for index, name in enumerate(names):
        cell = post.cells[name]
        configured_dsp(cell)
        for port, direction in cell.directions.items():
            if direction == "input":
                outputs |= _names({f"dsp_input_{index:02d}_{port}": len(cell.connections[port])})
                count += len(cell.connections[port])
    if (
        details.get("physical_dsp_count") != len(names)
        or details.get("original_input_bits") != case.input_bits
        or details.get("original_output_bits") != case.data_width
        or details.get("compared_primitive_input_bits") != count
    ):
        raise ValueError(
            "The retained primitive comparison omitted original or connected primitive bits"
        )
    return AigerInterface(inputs, outputs)


def _completed_record(
    case: SynthesisCase, root: Path, tools: SynthesisToolchain
) -> dict[str, object]:
    """Read completed actual component evidence and derive its complete current binding."""
    directory = root / _PROOF_DIRECTORY
    widths = _widths(case)
    original = Netlist.read(directory / "original-word.json")
    reference = Netlist.read(directory / "reference-word.json")
    pre = Netlist.read(directory / "pre-dsp.json")
    post = Netlist.read(directory / "post-dsp.json")
    physical = Netlist.read(root / "mapped.json")
    for module in [original, reference, pre, post, physical]:
        module.primary_inputs(widths, case.data_width)
    for name, expected in _FLOWS.items():
        if (directory / name).read_text(encoding="utf-8") != expected:
            raise ValueError("A retained original checkpoint preparation flow changed")
    if (directory / "unit.v").read_bytes() != (root / "unit.v").read_bytes():
        raise ValueError("A checkpoint preparation used a different original RTL source")
    for stem in ["pre-dsp", "post-dsp"]:
        tools.check_library_log(directory / (stem + "-preparation.log"))
    source_method = read_source_normalization(case, directory, tools)
    word = _object(directory / "word-proof/word-composition.json")
    products = [(name, cell) for name, cell in pre.cells.items() if cell.kind == "$mul"]
    if word.get("products_proven") != len(products) or word.get(
        "unchanged_functions_checked"
    ) != len(pre.cells) - len(products):
        raise ValueError("A retained word proof omitted actual original function nodes")
    if word != prove_word_composition(pre, post, widths, case.data_width).to_dict():
        raise ValueError("The retained word proof differs from the actual complete word graphs")
    for index, (name, cell) in enumerate(products):
        product = _object(directory / f"word-proof/product-{index:03d}.json")
        if (
            product.get("original_cell") != name
            or product.get("complete_product_bits") != len(cell.connections["Y"])
            or product.get("difference") != []
        ):
            raise ValueError("A complete retained original multiplication was not proved")
    configurations = sorted(
        {configured_dsp(cell) for cell in post.cells.values() if cell.kind == "DSP48E1"}
    )
    contracts = [
        read_dsp_contract(
            configuration, tools, directory / f"contracts/contract-{index:02d}"
        ).to_dict()
        for index, configuration in enumerate(configurations)
    ]
    primitive_directory = directory / "primitive-proof"
    primitive = _object(primitive_directory / "primitive-residual.json")
    aligned = read_dsp_port_alignment(post, physical, primitive_directory)
    interface = _primitive_interface(case, aligned, physical, primitive)
    read_native_equivalence(primitive_directory, "word.aig", "physical.aig", interface, tools)
    if (
        (primitive_directory / "word.ys").read_text(encoding="utf-8") != WORD_EXPORT_FLOW
        or (primitive_directory / "word-abc.script").read_text(encoding="utf-8") != WORD_EXPORT_ABC
        or any(
            not cell.kind.startswith("$")
            for cell in Netlist.read(primitive_directory / "word-cut.json").cells.values()
        )
    ):
        raise ValueError("The retained generic word export flow or actual cell domain changed")
    tools.check_library_log(primitive_directory / "physical-synthesis.log")
    return {
        "schema_version": "sc-neurocore.function-composition.v1",
        "case": case.key,
        "status": "equivalent",
        "method": "compositional",
        "input_bits": case.input_bits,
        "output_bits": case.data_width,
        "latches": 0,
        "source_method": source_method,
        "word": word,
        "primitives": primitive,
        "contracts": contracts,
        "root_artifact_sha256": _root_bindings(case, root),
        "producer_sha256": _source_bindings(),
        "toolchain_sha256": tools.identity,
        "artifact_sha256": _artifacts(directory),
    }


@dataclass(frozen=True, slots=True)
class CompositionalMappingProof:
    """Retain the complete qualified mapping proof artifact closure.

    Parameters
    ----------
    artifact_sha256 : tuple of (str, str)
        Complete used original, checkpoint, native command and proof-file identities,
        relative to the original measurement directory.

    """

    artifact_sha256: tuple[tuple[str, str], ...]


def read_compositional_mapping_proof(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain
) -> CompositionalMappingProof:
    """Read a complete retained compositional proof without rerunning or restamping it.

    Parameters
    ----------
    case : SynthesisCase
        Exact original function, lowering and full numerical geometry.
    directory : pathlib.Path
        Retained original physical acquisition and completed composition-proof tree.
    tools : SynthesisToolchain
        Currently inspected native executables and complete technology data.

    Returns
    -------
    CompositionalMappingProof
        Original artifact identities after complete current consistency checks.
        This detects stale or damaged evidence, not coherent forgery of an entire
        owner-controlled acquisition and all its native records.

    Raises
    ------
    OSError, ValueError
        For changed source, tools, exports, component inputs, incomplete interfaces
        or any component without its actual positive qualified result.

    """
    tools.verify()
    proof_directory = directory / _PROOF_DIRECTORY
    record = proof_directory / "proof-record.json"
    if proof_directory.is_symlink() or record.is_symlink():
        raise ValueError(
            "A compositional proof must not follow a retained directory or record link"
        )
    expected = _completed_record(case, directory, tools)
    if record.read_text(encoding="utf-8") != _encoded(expected):
        raise ValueError(
            "Compositional proof artifacts differ from their qualified source-bound record"
        )
    root_bindings = _root_bindings(case, directory)
    artifacts = list(root_bindings.items())
    artifacts.extend(
        (f"{_PROOF_DIRECTORY}/{name}", value) for name, value in _artifacts(proof_directory).items()
    )
    artifacts.append((f"{_PROOF_DIRECTORY}/proof-record.json", digest(record)))
    return CompositionalMappingProof(tuple(artifacts))


def prove_compositional_mapping(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain, *, timeout: float
) -> CompositionalMappingProof:
    """Prove a complete actual DSP mapping through independently bound native checkpoints.

    Parameters
    ----------
    case : SynthesisCase
        Exact original function and independently driven argument geometry.
    directory : pathlib.Path
        Original completed physical JSON, Verilog and native statistics. Complete
        original ports remain present; physical mapping commands stay unchanged.
    tools : SynthesisToolchain
        Inspected existing native executables and complete technology libraries.
    timeout : float
        Positive caller deadline upper bound. Each native proof command is capped
        at ninety seconds, and native CEC never exceeds sixty seconds.

    Returns
    -------
    CompositionalMappingProof
        Complete original source, word identity, native primitive contract and
        all-input/full-result residual proof, after final retained-reader checks.

    Raises
    ------
    OSError, ValueError
        For changed inputs, unsupported operators, state, incomplete or negative
        components. Failed native acquisitions remain available for diagnosis.
    SynthesisToolError
        If an actual command fails or reaches its unchanged native ceiling.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("A compositional proof requires a finite positive native deadline")
    seconds = min(timeout, 90.0)
    before, code = _root_bindings(case, directory), _source_bindings()
    proof_directory = directory / _PROOF_DIRECTORY
    proof_directory.mkdir(parents=True, exist_ok=False)
    (proof_directory / "unit.v").write_bytes((directory / "unit.v").read_bytes())
    for name, flow in _FLOWS.items():
        _write(proof_directory / name, flow)
        log = proof_directory / (name.removesuffix(".ys") + "-preparation.log")
        tools.run([str(tools.yosys), "-Q", "-T", name], proof_directory, log, timeout=seconds)
        if name != "original-words.ys":
            tools.check_library_log(log)
    widths = _widths(case)
    prove_source_normalization(case, proof_directory, tools, timeout=seconds)
    pre, post = (
        Netlist.read(proof_directory / "pre-dsp.json"),
        Netlist.read(proof_directory / "post-dsp.json"),
    )
    prove_word_composition(pre, post, widths, case.data_width, proof_directory / "word-proof")
    configurations = sorted(
        {configured_dsp(cell) for cell in post.cells.values() if cell.kind == "DSP48E1"}
    )
    for index, configuration in enumerate(configurations):
        prove_dsp_contract(
            configuration,
            tools,
            proof_directory / f"contracts/contract-{index:02d}",
            timeout=seconds,
        )
    prove_primitive_residual(
        proof_directory / "post-dsp.json",
        directory / "mapped.json",
        widths,
        case.data_width,
        tools,
        proof_directory / "primitive-proof",
        timeout=seconds,
    )
    if before != _root_bindings(case, directory) or code != _source_bindings():
        raise ValueError("Original acquisition or proof implementation changed during composition")
    tools.verify()
    _write(
        proof_directory / "proof-record.json", _encoded(_completed_record(case, directory, tools))
    )
    return read_compositional_mapping_proof(case, directory, tools)
