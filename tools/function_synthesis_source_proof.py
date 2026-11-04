# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete original-source normalization proof

"""Require exact structural keys or a complete native original-domain proof."""

from __future__ import annotations

import gc
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_aiger_names,
    check_equivalent_verdict,
    prove_aiger_equivalence,
)
from tools.function_synthesis_gia import read_gia_equivalence
from tools.function_synthesis_netlist import Netlist, object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_structural_proof import (
    UnsupportedStructuralOperation,
    compare_word_structure,
)
from tools.function_synthesis_toolchain import SynthesisToolchain

SOURCE_AIG_FLOWS = {
    stem: (
        f"read_json {source}\n"
        "hierarchy -check -top function_unit\ntechmap\nopt_clean\n"
        "aigmap\nopt_clean\ncheck -assert\n"
        f"write_aiger -symbols {stem}.aig\n"
    )
    for stem, source in (
        ("reference", "original-word.json"),
        ("candidate", "reference-word.json"),
    )
}


def _widths(case: SynthesisCase) -> dict[str, int]:
    """Keep every original independently driven function argument."""
    return {
        f"argument{index}": case.data_width for index in range(case.input_bits // case.data_width)
    }


def _interface(case: SynthesisCase) -> AigerInterface:
    """Name every original input and output bit without narrowing the domain."""

    def names(widths: dict[str, int]) -> frozenset[str]:
        """Preserve scalar and vector names in the complete exported interface."""
        return frozenset(
            f"{name}[{index}]" for name, width in widths.items() for index in range(width)
        )

    return AigerInterface(names(_widths(case)), names({"result": case.data_width}))


def read_native_equivalence(
    directory: Path,
    reference: str,
    candidate: str,
    interface: AigerInterface,
    tools: SynthesisToolchain,
) -> None:
    """Validate complete retained interfaces and the actual positive native verdict.

    Parameters
    ----------
    directory : pathlib.Path
        Retained complete native proof artifacts.
    reference, candidate : str
        Original local AIGER export names.
    interface : AigerInterface
        All independently driven original input and complete output names.
    tools : SynthesisToolchain
        Inspected native executables and technology libraries.

    Raises
    ------
    OSError, ValueError
        For changed geometry, names or artifacts, or a missing decisive proof.

    """
    for name, label in [(reference, "reference"), (candidate, "candidate")]:
        check_aiger_header(directory / name, interface)
        check_aiger_names(directory / (label + "-io.log"), interface)
    if (directory / "gia-proof.json").exists() or (directory / "gia-proof.json").is_symlink():
        read_gia_equivalence(
            directory / reference, directory / candidate, interface, tools, directory
        )
    else:
        check_equivalent_verdict(directory / "equivalence.log")


def prove_source_normalization(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain, *, timeout: float
) -> None:
    """Prove the complete original function against its actual normalization checkpoint.

    Parameters
    ----------
    case : SynthesisCase
        Original independent input and complete result geometry.
    directory : pathlib.Path
        Fresh actual original-word.json and reference-word.json checkpoints.
    tools : SynthesisToolchain
        Inspected existing native tools.
    timeout : float
        Admitted finite per-command native ceiling, at most ninety seconds.
        Independent source exports run concurrently, with at most two native processes.

    Raises
    ------
    OSError, ValueError
        For malformed graphs, exceeded structural budgets or an unproved native
        fallback. Unsupported structure alone never becomes positive evidence.
    SynthesisToolError
        For an actual failed or bounded-out native command.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("A source proof requires a finite native deadline at most ninety seconds")
    original = Netlist.read(directory / "original-word.json")
    reference = Netlist.read(directory / "reference-word.json")
    structural = directory / "source-structure"
    try:
        comparison = compare_word_structure(
            original, reference, _widths(case), case.data_width, structural
        )
    except UnsupportedStructuralOperation as error:
        structural.mkdir()
        (structural / "structural-comparison.json").write_text(
            json.dumps(
                {
                    "all_output_function_keys_identical": None,
                    "structural_interpretation": "unsupported",
                    "unsupported_operation": error.operation,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
    else:
        if comparison.matches:
            del original, reference
            gc.collect()
            return
    del original, reference
    gc.collect()
    native = directory / "source-native"
    native.mkdir()
    for name in ("original-word.json", "reference-word.json"):
        (native / name).write_bytes((directory / name).read_bytes())
    for stem, flow in SOURCE_AIG_FLOWS.items():
        (native / f"{stem}.ys").write_text(flow)
    with ThreadPoolExecutor(max_workers=2) as executor:
        exports = [
            executor.submit(
                tools.run,
                [str(tools.yosys), "-Q", "-T", f"{stem}.ys"],
                native,
                native / f"{stem}-synthesis.log",
                timeout=timeout,
            )
            for stem in SOURCE_AIG_FLOWS
        ]
        for export in exports:
            export.result()
    prove_aiger_equivalence(
        native / "reference.aig",
        native / "candidate.aig",
        _interface(case),
        tools,
        native,
        timeout=timeout,
    )


def read_source_normalization(
    case: SynthesisCase, directory: Path, tools: SynthesisToolchain
) -> Literal["structural", "complete-aiger"]:
    """Reconstruct source identity without rerunning native commands or restamping artifacts.

    Parameters
    ----------
    case : SynthesisCase
        Complete original function geometry.
    directory : pathlib.Path
        Retained actual word checkpoints and source proof.
    tools : SynthesisToolchain
        Currently inspected native tools.

    Returns
    -------
    str
        Structural identity or actual complete-domain native proof method.

    Raises
    ------
    OSError, ValueError
        For incomplete structural admission, changed original checkpoint data,
        or a native fallback without its actual positive full-domain result.

    """
    record = directory / "source-structure/structural-comparison.json"
    if record.is_symlink():
        raise ValueError("A retained source comparison must not be a symbolic link")
    structure = object_mapping(json.loads(record.read_bytes()))
    identical = structure.get("all_output_function_keys_identical")
    if identical is True:
        comparison = compare_word_structure(
            Netlist.read(directory / "original-word.json"),
            Netlist.read(directory / "reference-word.json"),
            _widths(case),
            case.data_width,
        )
        if not comparison.matches or structure != comparison.to_dict():
            raise ValueError("A retained structural proof differs from the actual source graphs")
        return "structural"
    if identical is not False:
        operation = structure.get("unsupported_operation")
        present = {
            cell.kind
            for name in ("original-word.json", "reference-word.json")
            for cell in Netlist.read(directory / name).cells.values()
        }
        if (
            identical is not None
            or set(structure)
            != {
                "all_output_function_keys_identical",
                "structural_interpretation",
                "unsupported_operation",
            }
            or structure.get("structural_interpretation") != "unsupported"
            or not isinstance(operation, str)
            or operation not in present
        ):
            raise ValueError("The original word structural comparison has no completed result")
    native = directory / "source-native"
    if any(
        (native / f"{stem}.ys").read_text() != flow for stem, flow in SOURCE_AIG_FLOWS.items()
    ) or any(
        (native / name).read_bytes() != (directory / name).read_bytes()
        for name in ("reference-word.json", "original-word.json")
    ):
        raise ValueError("The native source proof used different original or checkpoint exports")
    read_native_equivalence(native, "reference.aig", "candidate.aig", _interface(case), tools)
    return "complete-aiger"
