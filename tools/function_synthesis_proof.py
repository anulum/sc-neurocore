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
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import digest
from tools.function_synthesis_dsp_contracts import prove_dsp_contract, read_dsp_contract
from tools.function_synthesis_exports import MAPPING_FLOW, SYNTHESIS_FLOW, module_json_flow
from tools.function_synthesis_json import native_netlist_artifacts
from tools.function_synthesis_mapping_stages import prove_mapping_stages, read_mapping_stages
from tools.function_synthesis_netlist import Netlist, object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_residual_exports import export_pre_lut, verify_pre_lut_exports
from tools.function_synthesis_source_proof import (
    prove_source_normalization,
    read_source_normalization,
)
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_word_graph import configured_dsp
from tools.function_synthesis_word_groups import GroupedWordComposition
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
        "word_groups",
        "structural_proof",
        "aiger",
        "dsp_proof",
        "residual_exports",
        "checkpoints",
        "checkpoint_timeout",
        "mapping_stages",
        "dsp_ports",
        "dsp_contracts",
        "proof",
        "toolchain",
        "records",
        "yosys",
        "exports",
        "gia",
        "gia_partition",
        "gia_protocol",
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


def _completed_record(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    *,
    word_work: list[dict[str, object]],
    initial: bool = False,
) -> dict[str, object]:
    """Read completed actual component evidence and derive its complete current binding."""
    directory = root / _PROOF_DIRECTORY
    widths = _widths(case)
    original = Netlist.read(directory / "original-word.json")
    reference = Netlist.read(directory / "reference-word.json")
    pre = Netlist.read(directory / "pre-dsp.json")
    post = Netlist.read(directory / "post-dsp.json")
    physical = Netlist.read(root / "mapped.json")
    intermediate = Netlist.read(directory / "pre-lut.json")
    for module in [original, reference, pre, post, intermediate, physical]:
        module.primary_inputs(widths, case.data_width)
    for name, expected in _FLOWS.items():
        if (directory / name).read_text(encoding="utf-8") != expected:
            raise ValueError("A retained original checkpoint preparation flow changed")
    if (directory / "unit.v").read_bytes() != (root / "unit.v").read_bytes():
        raise ValueError("A checkpoint preparation used a different original RTL source")
    for stem in ("pre-dsp", "post-dsp"):
        tools.check_library_log(directory / (stem + "-preparation.log"))
    verify_pre_lut_exports(post, directory, tools)
    source_method = read_source_normalization(case, directory, tools)
    word = _object(directory / "word-proof/word-composition.json")
    products = [(name, cell) for name, cell in pre.cells.items() if cell.kind == "$mul"]
    grouped = word.get("schema_version") == "sc-neurocore.word-composition.v2"
    if not grouped and (
        word.get("products_proven") != len(products)
        or word.get("unchanged_functions_checked") != len(pre.cells) - len(products)
    ):
        raise ValueError("A retained word proof omitted actual original function nodes")
    actual_word = prove_word_composition(
        pre, post, widths, case.data_width, deadline=tools.cohort_deadline
    )
    if word != actual_word.to_dict():
        raise ValueError("The retained word proof differs from the actual complete word graphs")
    if isinstance(actual_word, GroupedWordComposition):
        observed = actual_word.work.to_dict()
        producer_work = _object(directory / "word-proof/group-work.json")
        producer_elapsed = actual_word.work.check_record(producer_work)
        reader_work = observed
        if not initial:
            retained_work = _object(directory / "word-readback-work.json")
            if retained_work.get("producer") != producer_work:
                raise ValueError("Retained grouped producer work differs from its original record")
            reader_work = object_mapping(retained_work.get("reader"))
        reader_elapsed = actual_word.work.check_record(reader_work)
        if producer_elapsed + reader_elapsed > 3300:
            raise ValueError("Retained grouped arithmetic work exhausted the shared case budget")
        word_work.append(observed)
    for index, (name, cell) in enumerate([] if grouped else products):
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
    mappings = read_mapping_stages(
        post, intermediate, physical, widths, case.data_width, tools, directory
    )
    return {
        "schema_version": "sc-neurocore.function-composition.v3",
        "case": case.key,
        "status": "equivalent",
        "method": "compositional",
        "input_bits": case.input_bits,
        "output_bits": case.data_width,
        "latches": 0,
        "source_method": source_method,
        "word": word,
        **mappings,
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
    word_work : tuple of dict, optional
        Actual current grouped reader work, kept separate from artifact identity.

    """

    artifact_sha256: tuple[tuple[str, str], ...]
    word_work: tuple[dict[str, object], ...] = field(default=(), compare=False)


def _seal_compositional_proof(
    case: SynthesisCase,
    directory: Path,
    tools: SynthesisToolchain,
    expected: dict[str, object],
    work: list[dict[str, object]],
    deadline: float,
) -> CompositionalMappingProof:
    """Check sealed bytes against an already reconstructed independent proof."""
    proof_directory = directory / _PROOF_DIRECTORY
    record = proof_directory / "proof-record.json"
    if proof_directory.is_symlink() or record.is_symlink():
        raise ValueError(
            "A compositional proof must not follow a retained directory or record link"
        )
    if record.read_text(encoding="utf-8") != _encoded(expected):
        raise ValueError(
            "Compositional proof artifacts differ from their qualified source-bound record"
        )
    root_bindings = _root_bindings(case, directory)
    proof_bindings = _artifacts(proof_directory)
    if (
        root_bindings != expected["root_artifact_sha256"]
        or proof_bindings != expected["artifact_sha256"]
        or _source_bindings() != expected["producer_sha256"]
    ):
        raise ValueError(
            "Compositional proof artifacts differ from their qualified source-bound record"
        )
    artifacts = list(root_bindings.items())
    artifacts.extend(
        (f"{_PROOF_DIRECTORY}/{name}", value) for name, value in proof_bindings.items()
    )
    artifacts.append((f"{_PROOF_DIRECTORY}/proof-record.json", digest(record)))
    tools.verify()
    if time.monotonic() > deadline:
        raise ValueError("Compositional proof reader exhausted the shared case budget")
    return CompositionalMappingProof(tuple(artifacts), tuple(work))


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
    deadline = time.monotonic() + 3300
    if tools.cohort_deadline is not None:
        deadline = min(deadline, tools.cohort_deadline)
    tools = replace(tools, cohort_deadline=deadline)
    tools.verify()
    proof_directory = directory / _PROOF_DIRECTORY
    record = proof_directory / "proof-record.json"
    if proof_directory.is_symlink() or record.is_symlink():
        raise ValueError(
            "A compositional proof must not follow a retained directory or record link"
        )
    work: list[dict[str, object]] = []
    expected = _completed_record(case, directory, tools, word_work=work)
    return _seal_compositional_proof(case, directory, tools, expected, work, deadline)


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
    deadline = time.monotonic() + 3300
    if tools.cohort_deadline is not None:
        deadline = min(deadline, tools.cohort_deadline)
    tools = replace(tools, cohort_deadline=deadline)
    before, code = _root_bindings(case, directory), _source_bindings()
    proof_directory = directory / _PROOF_DIRECTORY
    proof_directory.mkdir(parents=True, exist_ok=False)
    (proof_directory / "unit.v").write_bytes((directory / "unit.v").read_bytes())
    for name, flow in _FLOWS.items():
        _write(proof_directory / name, flow)
        stem = name.removesuffix(".ys")
        log = proof_directory / (stem + "-preparation.log")
        tools.run([str(tools.yosys), "-Q", "-T", "-s", name], proof_directory, log, timeout=seconds)
        if name != "original-words.ys":
            tools.check_library_log(log)
    widths = _widths(case)
    prove_source_normalization(case, proof_directory, tools, timeout=seconds)
    pre, post = (
        Netlist.read(proof_directory / "pre-dsp.json"),
        Netlist.read(proof_directory / "post-dsp.json"),
    )
    export_pre_lut(post, proof_directory, tools, seconds)
    prove_word_composition(
        pre,
        post,
        widths,
        case.data_width,
        proof_directory / "word-proof",
        deadline=tools.cohort_deadline,
    )
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
    prove_mapping_stages(
        proof_directory / "post-dsp.json",
        proof_directory / "pre-lut.json",
        directory / "mapped.json",
        widths,
        case.data_width,
        tools,
        proof_directory,
        timeout=seconds,
    )
    if before != _root_bindings(case, directory) or code != _source_bindings():
        raise ValueError("Original acquisition or proof implementation changed during composition")
    tools.verify()
    reader_work: list[dict[str, object]] = []
    payload = _completed_record(case, directory, tools, word_work=reader_work, initial=True)
    if reader_work:
        _write(
            proof_directory / "word-readback-work.json",
            _encoded(
                {
                    "producer": _object(proof_directory / "word-proof/group-work.json"),
                    "reader": reader_work[0],
                }
            ),
        )
        payload["artifact_sha256"] = _artifacts(proof_directory)
    _write(proof_directory / "proof-record.json", _encoded(payload))
    return _seal_compositional_proof(case, directory, tools, payload, reader_work, deadline)
