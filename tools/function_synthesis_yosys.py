# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound function mapping and complete-output equivalence

"""Map standalone RTL and require explicit equivalence of every input/output bit."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import (
    AigerInterface,
    check_equivalent_verdict,
    read_aiger_port_order,
)
from tools.function_synthesis_bdd import prove_complete_bdd_mapping, read_complete_bdd_mapping
from tools.function_synthesis_exports import (
    EXPORT_ARTIFACTS,
    MAPPING_FLOW,
    export_complete_aigers,
    validate_complete_aiger_exports,
)
from tools.function_synthesis_exports import (
    SYNTHESIS_FLOW as SYNTHESIS_FLOW,
)
from tools.function_synthesis_json import (
    MAX_NETLIST_BYTES,
    native_netlist_artifacts,
    normalize_native_netlist,
)
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_proof import (
    prove_compositional_mapping,
    read_compositional_mapping_proof,
)
from tools.function_synthesis_records import (
    SynthesisCase,
    SynthesisResources,
    declared_table_bits,
    read_synthesis_resources,
)
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

_ARTIFACTS = (
    "unit.v",
    "flow.ys",
    "mapped.json",
    "mapped.v",
    "stat.json",
    "reference.aig",
    "mapped.aig",
    "yosys.log",
    "yosys.command.json",
    "reference-io.log",
    "reference-io.command.json",
    "mapped-io.log",
    "mapped-io.command.json",
    "equivalence.log",
    "equivalence.command.json",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mapped_resources(case: SynthesisCase, directory: Path) -> SynthesisResources:
    """Match actual physical primitive counts and complete ports to the native statistics."""
    native_netlist_artifacts(directory)
    resources = read_synthesis_resources(directory / "stat.json")
    module = Netlist.read(directory / "mapped.json")
    module.primary_inputs(
        {
            f"argument{index}": case.data_width
            for index in range(case.input_bits // case.data_width)
        },
        case.data_width,
    )
    if dict(resources.cell_types) != dict(Counter(cell.kind for cell in module.cells.values())):
        raise ValueError("Mapped physical cells differ from the reported native statistics")
    return resources


def _check_sources(case: SynthesisCase, directory: Path) -> None:
    expected = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    if (directory / "unit.v").read_text(encoding="utf-8") != expected:
        raise ValueError("Reference RTL differs from the selected function")
    if (directory / "flow.ys").read_text(encoding="utf-8") not in {SYNTHESIS_FLOW, MAPPING_FLOW}:
        raise ValueError("Function synthesis flow differs from the bound script")


def _check_header(case: SynthesisCase, path: Path) -> None:
    with path.open("rb") as stream:
        line = stream.readline(256)
    fields = line.split()
    if (
        not line.endswith(b"\n")
        or len(fields) != 6
        or fields[0] != b"aig"
        or not all(field.isdigit() for field in fields[1:])
    ):
        raise ValueError("Expected a complete binary combinational AIGER header")
    maximum, inputs, latches, outputs, gates = map(int, fields[1:])
    if (inputs, latches, outputs) != (
        case.input_bits,
        0,
        case.data_width,
    ) or maximum != inputs + gates:
        raise ValueError("AIGER geometry differs from the full function interface")


def _check_interface(case: SynthesisCase, log: Path) -> None:
    """Keep complete named inputs and exact outputs, admitting native name-matched input permutations."""
    inputs = tuple(
        f"argument{index // case.data_width}[{index % case.data_width}]"
        for index in range(case.input_bits)
    )
    outputs = tuple(f"result[{index}]" for index in range(case.data_width))
    interface = AigerInterface(frozenset(inputs), frozenset(outputs))
    try:
        actual_inputs, actual_outputs = read_aiger_port_order(log, interface)
    except ValueError as error:
        raise ValueError("ABC did not confirm the complete named function interface") from error
    if actual_outputs != outputs or (
        actual_inputs != inputs and "[default = by name]" not in log.read_text(encoding="utf-8")
    ):
        raise ValueError("ABC did not confirm the complete named function interface")


def check_mapped_equivalence(
    case: SynthesisCase,
    directory: Path,
    tools: SynthesisToolchain,
    *,
    timeout: float = 90.0,
    cohort_seconds: float = 3300.0,
) -> None:
    """Require full-port equivalence of the two retained synthesis exports.

    Parameters
    ----------
    case : SynthesisCase
        Function and full signed input/output geometry.
    directory : pathlib.Path
        Exact original RTL, bound physical flow and actual JSON/statistics.
        DSP-bearing acquisitions first request compositional checkpoint evidence.
        If that attempt is inconclusive or refused, a direct full-original-domain
        native BDD proof is required instead; failed component artifacts remain.
        Both attempts must use the same original acquisition and implementation.
        Other mappings and legacy bare export pairs require complete original
        reference.aig/mapped.aig exports and direct full-domain native equality.
    tools : SynthesisToolchain
        Inspected executables and installed library bindings.
    timeout : float, optional
        Positive deadline upper bound for each native interface or equivalence
        check. Actual native proof commands never exceed ninety seconds.
    cohort_seconds : float, optional
        Finite positive overall budget, at most 3300 seconds, including source
        checks, composition, complete-domain fallback and final validation.
        A fallback receives only the remaining budget; every native command
        must fit in the same shared deadline before it can start.

    Raises
    ------
    OSError, ValueError
        For changed source/export bytes, malformed ports or any non-positive,
        conflicting, missing or inconclusive equivalence verdict.
    SynthesisToolError
        If a native command fails or reaches its elapsed deadline.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("A mapped proof requires a finite positive native deadline")
    if (
        isinstance(cohort_seconds, bool)
        or not math.isfinite(cohort_seconds)
        or not 0 < cohort_seconds <= 3300
    ):
        raise ValueError("A mapped proof requires a finite positive bounded cohort")
    deadline = time.monotonic() + cohort_seconds
    if tools.cohort_deadline is not None:
        if not math.isfinite(tools.cohort_deadline):
            raise ValueError("A mapped proof requires a finite inherited cohort deadline")
        deadline = min(deadline, tools.cohort_deadline)
    tools = replace(tools, cohort_deadline=deadline)
    timeout = min(timeout, 90.0)
    _check_sources(case, directory)
    resources = _mapped_resources(case, directory) if (directory / "mapped.json").exists() else None
    if resources is not None and resources.dsp_blocks:
        inputs = (
            *(directory / name for name in (*_ARTIFACTS[:5], "yosys.log", "yosys.command.json")),
            *(directory / name for name in native_netlist_artifacts(directory)),
            *Path(__file__).parent.glob("function_synthesis*.py"),
        )
        before_mapping = {path: (_digest(path), path.is_symlink()) for path in inputs}
        try:
            prove_compositional_mapping(case, directory, tools, timeout=timeout)
        except (ValueError, SynthesisToolError):
            if before_mapping != {path: (_digest(path), path.is_symlink()) for path in inputs}:
                raise ValueError(
                    "Original acquisition or proof implementation changed during composition"
                ) from None
            prove_complete_bdd_mapping(
                case,
                directory,
                tools,
                timeout=timeout,
                cohort_seconds=deadline - time.monotonic(),
            )
    else:
        exports = ("reference.aig", "mapped.aig")
        before = {name: _digest(directory / name) for name in exports}
        for name, stem in zip(exports, ("reference", "mapped"), strict=True):
            _check_header(case, directory / name)
            log = directory / f"{stem}-io.log"
            tools.run(
                [str(tools.abc), "-c", f"read_aiger {name}; print_io; cec -h"],
                directory,
                log,
                timeout=timeout,
            )
            _check_interface(case, log)
        proof_log = directory / "equivalence.log"
        tools.run(
            [
                str(tools.abc),
                "-c",
                f"cec -T {max(1, min(60, int(timeout)))} -C 1000000 reference.aig mapped.aig",
            ],
            directory,
            proof_log,
            timeout=timeout,
        )
        check_equivalent_verdict(proof_log)
        if before != {name: _digest(directory / name) for name in exports}:
            raise ValueError("AIGER exports changed during their equivalence check")
    _check_sources(case, directory)
    if time.monotonic() > deadline:
        raise ValueError("Mapped proof validation exhausted the shared cohort budget")


@dataclass(frozen=True, slots=True)
class FunctionSynthesisMeasurement:
    """Hold one actual, source-bound mapping with a successful complete proof.

    Parameters
    ----------
    case : SynthesisCase
        Function, lowering and signed numerical format.
    resources : SynthesisResources
        Actual mapped top-module statistics.
    table_bits : int
        Declared source table bits, separately from physical resource counts.
    toolchain_identity : str
        Canonical executable/library manifest digest.
    artifact_sha256 : tuple of (str, str)
        Exact input, flow, netlist, interface and proof-file digests.
    proof_method : str, optional
        Actual complete proof path used for this measurement. Compositional
        proofs retain native checkpoint, arithmetic-contract and residual evidence.
        Complete BDD proofs compare the whole original function directly, without
        auxiliary primitive cuts.

    """

    case: SynthesisCase
    resources: SynthesisResources
    table_bits: int
    toolchain_identity: str
    artifact_sha256: tuple[tuple[str, str], ...]
    proof_method: Literal[
        "complete-aiger", "compositional", "complete-bdd", "complete-bdd-shannon"
    ] = "complete-aiger"

    def to_dict(self) -> dict[str, object]:
        """Return the measured area and its explicit complete-proof context.

        Returns
        -------
        dict of str to object
            Actual resource/provenance values. Domain and accuracy measurements
            are added separately; equal format does not mean equal error.

        """
        return {
            "function": self.case.function,
            "lowering": self.case.lowering,
            "data_width": self.case.data_width,
            "fraction": self.case.fraction,
            "resources": self.resources.to_dict(),
            "declared_table_bits": self.table_bits,
            "latency_cycles": 0,
            "proof": {
                "status": "equivalent",
                "method": self.proof_method,
                "input_bits": self.case.input_bits,
                "output_bits": self.case.data_width,
                "latches": 0,
            },
            "toolchain_sha256": self.toolchain_identity,
            "artifact_sha256": dict(self.artifact_sha256),
        }


def read_function_synthesis_measurement(
    case: SynthesisCase,
    directory: Path,
    tools: SynthesisToolchain,
) -> FunctionSynthesisMeasurement:
    """Reconstruct retained measurements without claiming a fresh proof execution.

    Parameters
    ----------
    case : SynthesisCase
        Exact supported function, lowering and numerical format.
    directory : pathlib.Path
        Retained real mapping, full-interface and successful proof artifacts.
    tools : SynthesisToolchain
        Currently verified executables and installed technology libraries.

    Returns
    -------
    FunctionSynthesisMeasurement
        Counts and current artifact digests after the same source, geometry,
        interface and explicit-verdict admission used by actual execution.
        Cache admission additionally matches these digests to its saved record.

    Raises
    ------
    OSError, ValueError
        For missing, linked, changed or unqualified inputs and tool artifacts.

    """
    tools.verify()
    if any((directory / name).is_symlink() for name in _ARTIFACTS):
        raise ValueError("Retained measurement artifacts must not be symbolic links")
    _check_sources(case, directory)
    tools.check_library_log(directory / "yosys.log")
    resources = _mapped_resources(case, directory)
    proof_method: Literal["complete-aiger", "compositional", "complete-bdd", "complete-bdd-shannon"]
    if (directory / "complete-bdd-proof/proof-record.json").exists():
        complete = read_complete_bdd_mapping(case, directory, tools)
        artifacts = complete.artifact_sha256
        proof_method = complete.method
    elif resources.dsp_blocks:
        composed = read_compositional_mapping_proof(case, directory, tools)
        artifacts = composed.artifact_sha256
        proof_method = "compositional"
    else:
        for name in ("reference", "mapped"):
            _check_header(case, directory / f"{name}.aig")
            _check_interface(case, directory / f"{name}-io.log")
        extra: tuple[str, ...] = ()
        if (directory / "flow.ys").read_text(encoding="utf-8") == MAPPING_FLOW:
            validate_complete_aiger_exports(case, directory, tools)
            extra = EXPORT_ARTIFACTS
        check_equivalent_verdict(directory / "equivalence.log")
        artifacts = tuple(
            (name, _digest(directory / name))
            for name in (*_ARTIFACTS, *extra, *native_netlist_artifacts(directory))
        )
        proof_method = "complete-aiger"
    return FunctionSynthesisMeasurement(
        case,
        resources,
        declared_table_bits(case),
        tools.identity,
        artifacts,
        proof_method,
    )


def run_function_synthesis(
    case: SynthesisCase,
    directory: Path,
    tools: SynthesisToolchain,
    *,
    synthesis_timeout: float = 240.0,
    proof_timeout: float = 90.0,
) -> FunctionSynthesisMeasurement:
    """Generate, map and prove one complete standalone function measurement.

    Parameters
    ----------
    case : SynthesisCase
        A function implemented by the selected lowering.
    directory : pathlib.Path
        New private working directory, never a reused or partially successful
        attempt. The caller supplies the CPU and memory containment envelope.
    tools : SynthesisToolchain
        Existing inspected and source-bound tools.
    synthesis_timeout, proof_timeout : float, optional
        Finite positive native-command deadlines in seconds. Native proof
        commands are capped at ninety seconds, including larger caller budgets.

    Returns
    -------
    FunctionSynthesisMeasurement
        Actual xc7 cell/DSP counts and source table bits only after complete
        mapped equivalence. No physical timing, power or accuracy claim.

    Raises
    ------
    OSError, ValueError
        For unsupported functions, existing directories, invalid artifacts,
        source changes or an unproved mapping.
    SynthesisToolError
        If a tool fails or a command reaches its deadline.

    """
    if not case.supported:
        raise ValueError(f"Unsupported function synthesis case: {case.key}")
    tools.verify()
    directory.mkdir(parents=True, exist_ok=False)
    rtl = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    (directory / "unit.v").write_text(rtl, encoding="utf-8")
    (directory / "flow.ys").write_text(MAPPING_FLOW, encoding="utf-8")
    tools.run(
        [str(tools.yosys), "-Q", "-T", "-s", "flow.ys"],
        directory,
        directory / "yosys.log",
        timeout=synthesis_timeout,
    )
    tools.check_library_log(directory / "yosys.log")
    if (directory / "mapped.json").stat().st_size > MAX_NETLIST_BYTES:
        normalize_native_netlist(directory)
    resources = _mapped_resources(case, directory)
    if not resources.dsp_blocks:
        export_complete_aigers(case, directory, tools, timeout=proof_timeout)
    check_mapped_equivalence(case, directory, tools, timeout=proof_timeout)
    return read_function_synthesis_measurement(case, directory, tools)
