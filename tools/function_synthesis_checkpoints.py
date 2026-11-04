# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native function export checkpoints

"""Preserve full native function graphs across bounded lossless export steps."""

from __future__ import annotations

import gzip
import json
import math
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import AigerInterface, check_aiger_header, digest
from tools.function_synthesis_exports import MAPPING_FLOW, SYNTHESIS_FLOW
from tools.function_synthesis_json import native_netlist_artifacts
from tools.function_synthesis_netlist import Netlist, object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain

_RECORD = "checkpoint-exports.json"
_ROOT = (
    "unit.v",
    "flow.ys",
    "mapped.json",
    "mapped.v",
    "stat.json",
    "yosys.log",
    "yosys.command.json",
)
_PREPARE = "hierarchy -check -top function_unit\nproc\nmemory_map\nopt\nwreduce\nopt\n"
_MAX_CHECKPOINT_BYTES = 1024**3


def _encoded(value: object) -> str:
    """Serialize one original native acquisition without ambiguous key ordering."""
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _regular(path: Path) -> None:
    """Refuse links and special entries before reading retained native inputs."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Complete function exports require regular unlinked artifacts")


def _sources() -> dict[str, str]:
    """Bind every actual export producer, graph reader and native command dependency."""
    return {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "function_synthesis_checkpoints.py",
            "function_synthesis_exports.py",
            "function_synthesis_aiger.py",
            "function_synthesis_json.py",
            "function_synthesis_netlist.py",
            "function_synthesis_records.py",
            "function_synthesis_toolchain.py",
        )
    }


def _interface(case: SynthesisCase) -> AigerInterface:
    """Keep every original argument independent and every original result bit."""
    return AigerInterface(
        frozenset(
            f"argument{argument}[{bit}]"
            for argument in range(case.input_bits // case.data_width)
            for bit in range(case.data_width)
        ),
        frozenset(f"result[{bit}]" for bit in range(case.data_width)),
    )


def _root(case: SynthesisCase, directory: Path) -> tuple[dict[str, str], str, str]:
    """Reconstruct the exact original RTL and complete physical module."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("A complete function acquisition requires a regular directory")
    names = (*_ROOT, *native_netlist_artifacts(directory))
    for name in names:
        _regular(directory / name)
    rtl = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    if (directory / "unit.v").read_text(encoding="utf-8") != rtl or (
        directory / "flow.ys"
    ).read_text(encoding="utf-8") not in {MAPPING_FLOW, SYNTHESIS_FLOW}:
        raise ValueError("The complete reference or original physical mapping flow changed")
    physical = Netlist.read(directory / "mapped.json")
    physical.primary_inputs(
        {
            f"argument{index}": case.data_width
            for index in range(case.input_bits // case.data_width)
        },
        case.data_width,
    )
    return (
        {name: digest(directory / name) for name in names},
        rtl,
        _encoded({"modules": {"function_unit": physical.clone_module()}}),
    )


def _flows() -> dict[str, str]:
    """Describe every complete native transformation and checkpoint readback step."""
    result = {}
    for role in ("reference", "mapped"):
        start = (
            "read_verilog -sv unit.v\n" + _PREPARE
            if role == "reference"
            else (
                "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
                "read_json physical-module.json\nhierarchy -check -top function_unit\n"
                "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v\n"
                "proc\nmemory_map\nopt\nwreduce\nopt\n"
            )
        )
        result[role + "-prepare.ys"] = start + f"write_rtlil {role}.prepared.il.gz\n"
        result[role + "-map.ys"] = (
            f"read_rtlil {role}.prepared.il.gz\nhierarchy -check -top function_unit\n"
            "techmap\nopt_clean\nselect -module function_unit\nrename -enumerate\n"
            f"write_rtlil -selected {role}.il.gz\n"
        )
        result[role + "-finish.ys"] = (
            f"read_rtlil {role}.il.gz\nhierarchy -check -top function_unit\n"
            "aigmap\nopt_clean\ncheck -assert\n"
            f"write_aiger -symbols {role}.aig\n"
        )
    return result


def _checkpoint(path: Path) -> dict[str, object]:
    """Verify the whole gzip stream, including its checksum and bounded decoded volume."""
    size = 0
    try:
        with gzip.open(path, "rb") as stream:
            while data := stream.read(65536):
                size += len(data)
                if size > _MAX_CHECKPOINT_BYTES:
                    raise ValueError(
                        "The complete decoded native checkpoint exceeds its byte budget"
                    )
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("The retained native checkpoint is truncated or corrupt") from error
    if not size:
        raise ValueError("A native function checkpoint must not be empty")
    return {"compressed_sha256": digest(path), "decoded_bytes": size}


def _command(directory: Path, stem: str, tools: SynthesisToolchain) -> None:
    """Require the inspected executable and bounded original native command."""
    value: object = json.loads((directory / (stem + ".command.json")).read_text())
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("A native checkpoint command requires its original argument vector")
    argv = [item for item in value if isinstance(item, str)]
    if (
        len(argv) != 12
        or argv[:6]
        != [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
        ]
        or argv[7:] != [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"]
    ):
        raise ValueError("The retained checkpoint command differs from the bounded native export")
    seconds = float(argv[6])
    if not math.isfinite(seconds) or not 0 < seconds <= 90:
        raise ValueError("The retained checkpoint command exceeds its native deadline")


def _completed(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    directory: Path,
    timestamp: str,
) -> dict[str, object]:
    """Recompute every original input, export script and actual retained artifact binding."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Complete native exports require a regular proof directory")
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo != UTC or parsed.isoformat() != timestamp:
        raise ValueError("Native exports require the original canonical UTC timestamp")
    inputs, rtl, physical = _root(case, root)
    expected = {"unit.v": rtl, "physical-module.json": physical, **_flows()}
    names = set(expected)
    for role in ("reference", "mapped"):
        names.update({role + ".prepared.il.gz", role + ".il.gz", role + ".aig"})
        for phase in ("prepare", "map", "finish"):
            stem = role + "-" + phase
            names.update({stem + ".log", stem + ".command.json"})
    for name in names:
        _regular(directory / name)
    if any(
        (directory / name).read_text(encoding="utf-8") != text for name, text in expected.items()
    ):
        raise ValueError(
            "The original native source, physical clone or complete export script changed"
        )
    checkpoints = {}
    prepared_checkpoints = {}
    for role in ("reference", "mapped"):
        for phase in ("prepare", "map", "finish"):
            _command(directory, role + "-" + phase, tools)
        check_aiger_header(directory / (role + ".aig"), _interface(case))
        prepared_checkpoints[role] = _checkpoint(directory / (role + ".prepared.il.gz"))
        checkpoints[role] = _checkpoint(directory / (role + ".il.gz"))
    tools.check_library_log(directory / "mapped-prepare.log")
    return {
        "schema_version": "sc-neurocore.complete-function-checkpoints.v2",
        "case": case.key,
        "recorded_at": timestamp,
        "original_input_bits": case.input_bits,
        "original_output_bits": case.data_width,
        "root_artifact_sha256": inputs,
        "producer_sha256": _sources(),
        "toolchain_sha256": tools.identity,
        "checkpoints": checkpoints,
        "prepared_checkpoints": prepared_checkpoints,
        "artifact_sha256": {name: digest(directory / name) for name in sorted(names)},
    }


def read_checkpoint_aigers(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    directory: Path,
) -> None:
    """Validate complete original native exports without execution or restamping.

    Parameters
    ----------
    case : SynthesisCase
        Original function and complete independently driven argument geometry.
    root : pathlib.Path
        Retained unchanged physical synthesis acquisition.
    tools : SynthesisToolchain
        Currently inspected executables and complete technology library.
    directory : pathlib.Path
        Original checkpoint exports and their unmodified acquisition record.

    Raises
    ------
    OSError, ValueError
        For changed, linked, truncated, oversized or incomplete native evidence.

    """
    tools.verify()
    path = directory / _RECORD
    _regular(path)
    text = path.read_text(encoding="utf-8")
    record = object_mapping(json.loads(text))
    timestamp = record.get("recorded_at")
    if not isinstance(timestamp, str):
        raise ValueError("Native checkpoint exports lack their original acquisition time")
    if text != _encoded(_completed(case, root, tools, directory, timestamp)):
        raise ValueError("The complete native export record differs from its original bindings")


def export_checkpoint_aigers(
    case: SynthesisCase,
    root: Path,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float = 90.0,
    cohort_seconds: float = 3300.0,
) -> None:
    """Export entire original functions through bounded native checkpoint steps.

    Parameters
    ----------
    case : SynthesisCase
        Original function and signed input/output geometry.
    root : pathlib.Path
        Unchanged real physical synthesis acquisition.
    tools : SynthesisToolchain
        Inspected native executable and technology-library closure.
    directory : pathlib.Path
        Fresh proof directory; all failed native steps remain retained.
    timeout : float, optional
        Positive finite native command ceiling, at most ninety seconds.
    cohort_seconds : float, optional
        Remaining whole-export allowance, positive and at most 3,300 seconds.
        Each of the six native commands must fit the remaining allowance.

    Raises
    ------
    OSError, ValueError
        For invalid inputs, existing output, changed sources or incomplete exports.
    SynthesisToolError
        If a native transformation fails or reaches its unchanged deadline.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("Native checkpoint exports require an admitted finite deadline")
    if (
        isinstance(cohort_seconds, bool)
        or not math.isfinite(cohort_seconds)
        or not 0 < cohort_seconds <= 3300
    ):
        raise ValueError("Native checkpoint exports require a finite cohort allowance")
    began = time.monotonic()
    tools.verify()
    before, rtl, physical = _root(case, root)
    sources = _sources()
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in {"unit.v": rtl, "physical-module.json": physical, **_flows()}.items():
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
    for stem in (name.removesuffix(".ys") for name in _flows()):
        if time.monotonic() - began > cohort_seconds - timeout:
            raise ValueError("The complete native checkpoint cohort is incomplete")
        tools.run(
            [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"],
            directory,
            directory / (stem + ".log"),
            timeout=timeout,
        )
    if _root(case, root)[0] != before or _sources() != sources:
        raise ValueError("The original complete function acquisition changed during export")
    tools.verify()
    record = _completed(case, root, tools, directory, datetime.now(UTC).isoformat())
    if time.monotonic() - began > cohort_seconds:
        raise ValueError("The complete native checkpoint cohort is incomplete")
    with (directory / _RECORD).open("x", encoding="utf-8") as stream:
        stream.write(_encoded(record))
    read_checkpoint_aigers(case, root, tools, directory)
    if time.monotonic() - began > cohort_seconds:
        raise ValueError("The complete native checkpoint cohort is incomplete")
