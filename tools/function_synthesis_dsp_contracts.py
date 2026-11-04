# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native complete-domain DSP arithmetic contracts

"""Prove actual configured DSP arithmetic against the inspected native library model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_aiger_names,
    digest,
)
from tools.function_synthesis_dsp_ports import DSP_AUXILIARY_INPUTS, MacroConfiguration
from tools.function_synthesis_exports import module_json_flow
from tools.function_synthesis_toolchain import SynthesisToolchain

_INPUTS = (
    ("a", 30),
    ("b", 18),
    ("acin", 30),
    ("bcin", 18),
    ("c", 48),
    ("pcin", 48),
    *(("control_" + name.lower(), 1) for name in DSP_AUXILIARY_INPUTS),
)
_OUTPUTS = (("result", 48), ("cascade_a", 30), ("cascade_b", 18), ("cascade_p", 48))
_REGISTERS = (
    "ACASCREG",
    "ADREG",
    "ALUMODEREG",
    "AREG",
    "BCASCREG",
    "BREG",
    "CARRYINREG",
    "CARRYINSELREG",
    "CREG",
    "DREG",
    "INMODEREG",
    "MREG",
    "OPMODEREG",
    "PREG",
)
_HEADER = """// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Complete configured DSP comparison
"""
_WORD = "flatten\nproc\nopt\nmemory_map\nopt\nwreduce\nopt\n"
_BITS = "techmap\nopt_clean\naigmap\nopt_clean\ncheck -assert\n"
_FLOW = (
    "read_verilog -sv -D EQUIV +/xilinx/cells_sim.v\n"
    "read_verilog -sv contract.v\ndesign -save original\nhierarchy -check -top reference\n"
    + _WORD
    + _BITS
    + "write_aiger -symbols reference.aig\n"
    "design -load original\nhierarchy -check -top candidate\n"
    + _WORD
    + _BITS
    + "write_aiger -symbols candidate.aig\n"
    "design -load original\nread_verilog -sv miter.v\nhierarchy -check -top contract_miter\n"
    + _WORD
    + module_json_flow("paired-word.json", "contract_miter")
    + _BITS
    + "write_aiger -symbols miter.aig\n"
)
_ARTIFACTS = (
    "contract.v",
    "miter.v",
    "flow.ys",
    "reference.aig",
    "candidate.aig",
    "miter.aig",
    "paired-word.json",
    "synthesis.log",
    "synthesis.command.json",
    "reference-io.log",
    "reference-io.command.json",
    "candidate-io.log",
    "candidate-io.command.json",
    "miter-io.log",
    "miter-io.command.json",
    "equivalence.log",
    "equivalence.command.json",
)


def _interface(outputs: tuple[tuple[str, int], ...]) -> AigerInterface:
    """Require every actual original bit, including all unused operand and cascade bits."""
    return AigerInterface(
        frozenset(
            name if width == 1 else f"{name}[{index}]"
            for name, width in _INPUTS
            for index in range(width)
        ),
        frozenset(
            name if width == 1 else f"{name}[{index}]"
            for name, width in outputs
            for index in range(width)
        ),
    )


def _rtl(configuration: MacroConfiguration) -> str:
    """Generate exact full-input arithmetic and actual configured native DSP wrappers."""
    mode, a_input, b_input = configuration
    if (
        mode not in {"0000101", "0010101", "1010101", "0110101"}
        or a_input not in {"DIRECT", "CASCADE"}
        or b_input not in {"DIRECT", "CASCADE"}
    ):
        raise ValueError("The requested native DSP contract has an unsupported configuration")
    a, b = ("acin" if a_input == "CASCADE" else "a"), ("bcin" if b_input == "CASCADE" else "b")
    z = {
        "0000101": "48'sd0",
        "0010101": "$signed(pcin)",
        "1010101": "$signed(pcin) >>> 17",
        "0110101": "$signed(c)",
    }[mode]
    ports = ", ".join(
        f"input {name}" if width == 1 else f"input [{width - 1}:0] {name}"
        for name, width in _INPUTS
    )
    ports += ", " + ", ".join(f"output [{width - 1}:0] {name}" for name, width in _OUTPUTS)
    parameters = ", ".join(f".{name}(0)" for name in _REGISTERS)
    parameters += f', .A_INPUT("{a_input}"), .B_INPUT("{b_input}"), .USE_DPORT("FALSE"), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")'
    controls = ", ".join(f".{name}(control_{name.lower()})" for name in DSP_AUXILIARY_INPUTS)
    return (
        _HEADER
        + f"""
module reference({ports});
wire signed [42:0] product = $signed({a}[24:0]) * $signed({b});
wire signed [47:0] z = {z};
assign result = $signed(product) + z;
assign cascade_a = {a};
assign cascade_b = {b};
assign cascade_p = result;
endmodule
module candidate({ports});
DSP48E1 #({parameters}) actual_macro (
.A(a), .B(b), .ACIN(acin), .BCIN(bcin), .C(c), .D(25'd0), .PCIN(pcin),
.INMODE(5'd0), .OPMODE(7'b{mode}), .ALUMODE(4'd0), .CARRYINSEL(3'd0), .CARRYIN(1'b0),
{controls},
.P(result), .ACOUT(cascade_a), .BCOUT(cascade_b), .PCOUT(cascade_p));
endmodule
"""
    )


def _miter() -> str:
    """Share all 218 actual input wires before comparing all 144 output bits."""
    ports = ", ".join(
        f"input {name}" if width == 1 else f"input [{width - 1}:0] {name}"
        for name, width in _INPUTS
    )
    text = _HEADER + f"\nmodule contract_miter({ports}, output difference);\n"
    for prefix, module in [("gold", "reference"), ("gate", "candidate")]:
        text += "".join(f"wire [{width - 1}:0] {prefix}_{name};\n" for name, width in _OUTPUTS)
        connections = [f".{name}({name})" for name, _ in _INPUTS]
        connections.extend(f".{name}({prefix}_{name})" for name, _ in _OUTPUTS)
        text += f"{module} {prefix}(" + ",".join(connections) + ");\n"
    gold = ",".join("gold_" + name for name, _ in _OUTPUTS)
    gate = ",".join("gate_" + name for name, _ in _OUTPUTS)
    return text + "assign difference = |({" + gold + "} ^ {" + gate + "});\nendmodule\n"


def _check_unsatisfiable(path: Path) -> None:
    """Require one decisive native UNSAT result for the complete paired miter."""
    text = path.read_text(encoding="utf-8")
    if re.findall(r"^(UNSATISFIABLE|SATISFIABLE)\b.*$", text, re.M) != [
        "UNSATISFIABLE"
    ] or re.search(r"^\s*(?:ERROR|Error)\b", text, re.M):
        raise ValueError("The complete native DSP arithmetic contract was not proved")


@dataclass(frozen=True, slots=True)
class NativeDspContract:
    """Bind one complete configured arithmetic proof to its real source and tool bytes.

    Parameters
    ----------
    configuration : tuple of str
        Actual OPMODE, A_INPUT and B_INPUT selection.
    recorded_at : str
        Original canonical UTC qualification timestamp, unchanged by retained reads.
    toolchain_identity, specification_identity : str
        Actual tool/library manifest and current contract-generator source digests.
    artifacts : tuple of (str, str)
        Every original source, full export, command and native proof log digest.

    """

    configuration: MacroConfiguration
    recorded_at: str
    toolchain_identity: str
    specification_identity: str
    artifacts: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        """Return complete contract provenance without claiming physical-device validation."""
        return {
            "configuration": list(self.configuration),
            "recorded_at": self.recorded_at,
            "toolchain_sha256": self.toolchain_identity,
            "specification_sha256": self.specification_identity,
            "independent_input_bits": sum(width for _, width in _INPUTS),
            "independent_operand_bits": 192,
            "independent_auxiliary_control_bits": len(DSP_AUXILIARY_INPUTS),
            "compared_output_bits": 144,
            "status": "equivalent",
            "artifact_sha256": dict(self.artifacts),
        }


def read_dsp_contract(
    configuration: MacroConfiguration,
    tools: SynthesisToolchain,
    directory: Path,
) -> NativeDspContract:
    """Validate retained complete native contract artifacts without claiming a new run.

    Parameters
    ----------
    configuration : tuple of str
        Exact selected OPMODE, A_INPUT and B_INPUT.
    tools : SynthesisToolchain
        Currently inspected native executables and installed library contents.
    directory : pathlib.Path
        Original private contract directory with exact source and native records.

    Returns
    -------
    NativeDspContract
        The original qualification, only after complete source, interface, verdict,
        tool and artifact consistency checks.

    Raises
    ------
    OSError, ValueError
        For linked, damaged, changed, unsupported or inconclusive proof artifacts.

    """
    tools.verify()
    if directory.is_symlink() or any(
        (directory / name).is_symlink() for name in (*_ARTIFACTS, "contract-record.json")
    ):
        raise ValueError("Retained native DSP contracts must not contain linked artifacts")
    for name, expected in [
        ("contract.v", _rtl(configuration)),
        ("miter.v", _miter()),
        ("flow.ys", _FLOW),
    ]:
        if (directory / name).read_text(encoding="utf-8") != expected:
            raise ValueError(
                "A retained DSP contract differs from its complete original specification"
            )
    tools.check_library_log(directory / "synthesis.log")
    for name, interface in [
        ("reference", _interface(_OUTPUTS)),
        ("candidate", _interface(_OUTPUTS)),
        ("miter", _interface((("difference", 1),))),
    ]:
        check_aiger_header(directory / (name + ".aig"), interface)
        check_aiger_names(directory / (name + "-io.log"), interface)
    _check_unsatisfiable(directory / "equivalence.log")
    record: object = json.loads((directory / "contract-record.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("recorded_at"), str):
        raise ValueError("A retained DSP contract lacks its original qualification timestamp")
    timestamp = record["recorded_at"]
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo != UTC or parsed.isoformat() != timestamp:
        raise ValueError("A retained DSP contract timestamp must be canonical UTC")
    result = NativeDspContract(
        configuration,
        timestamp,
        tools.identity,
        digest(Path(__file__)),
        tuple((name, digest(directory / name)) for name in _ARTIFACTS),
    )
    if record != result.to_dict():
        raise ValueError("Retained DSP proof artifacts differ from their original qualification")
    return result


def prove_dsp_contract(
    configuration: MacroConfiguration,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
) -> NativeDspContract:
    """Prove actual configured arithmetic over all 218 inputs and all 144 output bits.

    Parameters
    ----------
    configuration : tuple of str
        Exact OPMODE, A_INPUT and B_INPUT selection needed by the actual netlist.
    tools : SynthesisToolchain
        Existing inspected native executables and installed library contents.
    directory : pathlib.Path
        New private directory retaining the entire actual proof execution.
    timeout : float
        Finite per-native-command deadline, at most ninety seconds.

    Returns
    -------
    NativeDspContract
        Complete source-bound native proof, never an undecided or sampled result.

    Raises
    ------
    OSError, ValueError
        For unsupported configurations, changed sources, invalid interfaces or a
        result without a decisive complete native UNSAT proof.
    SynthesisToolError
        If an actual command fails or reaches its unchanged deadline.

    """
    if isinstance(timeout, bool) or not 0 < timeout <= 90:
        raise ValueError("A native DSP contract requires an admitted finite deadline")
    sources = {"contract.v": _rtl(configuration), "miter.v": _miter(), "flow.ys": _FLOW}
    specification = digest(Path(__file__))
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in sources.items():
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    tools.run(
        [str(tools.yosys), "-Q", "-T", "flow.ys"],
        directory,
        directory / "synthesis.log",
        timeout=timeout,
    )
    tools.check_library_log(directory / "synthesis.log")
    for name, interface in [
        ("reference", _interface(_OUTPUTS)),
        ("candidate", _interface(_OUTPUTS)),
        ("miter", _interface((("difference", 1),))),
    ]:
        check_aiger_header(directory / (name + ".aig"), interface)
        log = directory / (name + "-io.log")
        tools.run(
            [str(tools.abc), "-c", f"read_aiger {name}.aig; print_io"],
            directory,
            log,
            timeout=min(timeout, 15),
        )
        check_aiger_names(log, interface)
    log = directory / "equivalence.log"
    tools.run(
        [str(tools.abc), "-c", "read_aiger miter.aig; sat -C 1000000"],
        directory,
        log,
        timeout=timeout,
    )
    _check_unsatisfiable(log)
    if specification != digest(Path(__file__)) or any(
        (directory / name).read_text(encoding="utf-8") != text for name, text in sources.items()
    ):
        raise ValueError("The actual DSP arithmetic specification changed during proof")
    tools.verify()
    result = NativeDspContract(
        configuration,
        datetime.now(UTC).isoformat(),
        tools.identity,
        specification,
        tuple((name, digest(directory / name)) for name in _ARTIFACTS),
    )
    with (directory / "contract-record.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
    return read_dsp_contract(configuration, tools, directory)
