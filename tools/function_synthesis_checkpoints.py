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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_aiger import AigerInterface, check_aiger_header, digest
from tools.function_synthesis_checkpoint_timeout import (
    preserve_checkpoint_timeout,
    read_checkpoint_timeout,
    validate_checkpoint_command,
)
from tools.function_synthesis_exports import (
    LARGE_LUT_EXPANSION,
    MAPPING_FLOW,
    SYNTHESIS_FLOW,
    complete_checkpoint_blif_flows,
    lut_gate_expansion,
    validate_blif_output_drivers,
)
from tools.function_synthesis_json import native_netlist_artifacts
from tools.function_synthesis_netlist import Netlist, object_mapping
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_retention import (
    compress_native_artifact,
    compressed_native_path,
    is_compressed_native,
    read_compressed_native,
)
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

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
            "function_synthesis_checkpoint_timeout.py",
            "function_synthesis_exports.py",
            "function_synthesis_aiger.py",
            "function_synthesis_json.py",
            "function_synthesis_netlist.py",
            "function_synthesis_records.py",
            "function_synthesis_retention.py",
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


def _root(case: SynthesisCase, directory: Path) -> tuple[dict[str, str], str, str, bool]:
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
        lut_gate_expansion(physical) >= LARGE_LUT_EXPANSION,
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


def _export_plan(streamed: bool) -> tuple[dict[str, str], dict[str, str]]:
    """Select the same complete native stages for acquisition and readback."""
    if streamed:
        return complete_checkpoint_blif_flows()
    return {name.removesuffix(".ys"): script for name, script in _flows().items()}, {}


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


def _streamed_checkpoint(path: Path) -> dict[str, object]:
    """Verify a complete nonempty native RTLIL handoff with the same byte bound."""
    if is_compressed_native(path):
        decoded = read_compressed_native(path)
        if not decoded.prefix.startswith(b"# Generated by Yosys"):
            raise ValueError("The complete native RTLIL checkpoint has invalid size or format")
        if not decoded.suffix.endswith(b"\nend\n"):
            raise ValueError("The complete native RTLIL checkpoint lacks its final module boundary")
        return {"native_sha256": decoded.sha256, "decoded_bytes": decoded.size}
    if path.suffix == ".gz":
        return _checkpoint(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        if not 0 < size <= _MAX_CHECKPOINT_BYTES or not stream.readline(256).startswith(
            b"# Generated by Yosys"
        ):
            raise ValueError("The complete native RTLIL checkpoint has invalid size or format")
        stream.seek(-min(size, 16), 2)
        if not stream.read().endswith(b"\nend\n"):
            raise ValueError("The complete native RTLIL checkpoint lacks its final module boundary")
    return {"native_sha256": digest(path), "decoded_bytes": size}


def _blif_script(role: str) -> str:
    """Convert the complete validated native BLIF to the same named gate interface."""
    return f"read_blif {role}.blif; strash; write_aiger -s {role}.aig"


def _command(directory: Path, stem: str, tools: SynthesisToolchain) -> None:
    """Require the inspected executable and bounded original native command."""
    validate_checkpoint_command(
        directory, stem, tools, [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"], 90
    )


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
    inputs, rtl, physical, streamed = _root(case, root)
    timeout_attempt = read_checkpoint_timeout(directory, _export_plan(False)[0], tools)
    if timeout_attempt is not None and streamed:
        raise ValueError("The original staged checkpoint never attempted a direct export")
    streamed = streamed or timeout_attempt is not None
    flows, steps = _export_plan(streamed)
    steps = {
        stem: name if name.endswith(".gz") else compressed_native_path(Path(name)).name
        for stem, name in steps.items()
    }
    expected = {"unit.v": rtl, "physical-module.json": physical}
    expected.update({stem + ".ys": script for stem, script in flows.items()})
    names = set(expected)
    if timeout_attempt is not None:
        names.update(path.relative_to(directory).as_posix() for path in timeout_attempt[1])
    for stem in flows:
        names.update({stem + ".log", stem + ".command.json"})
    for role in ("reference", "mapped"):
        names.add(role + ".aig")
        if streamed:
            names.update({role + ".blif", role + "-aiger.log", role + "-aiger.command.json"})
        else:
            names.update({role + ".prepared.il.gz", role + ".il.gz"})
    names.update(steps.values())
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
    for stem in flows:
        _command(directory, stem, tools)
    for role in ("reference", "mapped"):
        check_aiger_header(directory / (role + ".aig"), _interface(case))
        if streamed:
            validate_blif_output_drivers(
                directory / (role + ".blif"), _interface(case), _MAX_CHECKPOINT_BYTES
            )
            validate_checkpoint_command(
                directory,
                role + "-aiger",
                tools,
                [str(tools.abc), "-s", "-c", _blif_script(role)],
                60,
            )
            log = (directory / (role + "-read-blif.log")).read_text(encoding="utf-8")
            if "Found and reported 0 problems." not in log:
                raise ValueError("A complete native BLIF did not pass structural readback")
        else:
            prepared_checkpoints[role] = _checkpoint(directory / (role + ".prepared.il.gz"))
            checkpoints[role] = _checkpoint(directory / (role + ".il.gz"))
    if streamed:
        prepared_checkpoints = {
            role: _streamed_checkpoint(directory / steps[role + "-prepare"])
            for role in ("reference", "mapped")
        }
        checkpoints = {
            name: _streamed_checkpoint(directory / name)
            for stem, name in steps.items()
            if not stem.endswith("-prepare")
        }
    tools.check_library_log(directory / "mapped-prepare.log")
    return {
        "schema_version": (
            "sc-neurocore.complete-function-checkpoints.v4"
            if streamed
            else "sc-neurocore.complete-function-checkpoints.v2"
        ),
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
        **({"initial_timeout_stage": timeout_attempt[0]} if timeout_attempt else {}),
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
        Every native command must fit this remaining allowance.

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
    before, rtl, physical, streamed = _root(case, root)
    sources = _sources()
    directory.mkdir(parents=True, exist_ok=False)
    flows, steps = _export_plan(streamed)
    for name, text in (("unit.v", rtl), ("physical-module.json", physical)):
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(text)

    def write_flows() -> None:
        """Preserve the exact native commands before each export route starts."""
        for stem, script in flows.items():
            with (directory / (stem + ".ys")).open("x", encoding="utf-8") as stream:
                stream.write(script)

    write_flows()

    def run(stem: str, command: list[str], ceiling: float) -> None:
        """Require the remaining original cohort before admitting a full native step."""
        if time.monotonic() - began > cohort_seconds - ceiling:
            raise ValueError("The complete native checkpoint cohort is incomplete")
        tools.run(command, directory, directory / (stem + ".log"), timeout=ceiling)

    def yosys(stem: str) -> None:
        """Run a bounded original command and verify a produced intermediate immediately."""
        run(stem, [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"], timeout)
        if stem in steps:
            path = directory / steps[stem]
            _regular(path)
            _streamed_checkpoint(path)

    def role_exports(role: str) -> None:
        """Build one complete role before converting its independent checked BLIF."""
        for stem in flows:
            if stem.startswith(role + "-"):
                yosys(stem)
        validate_blif_output_drivers(
            directory / (role + ".blif"), _interface(case), _MAX_CHECKPOINT_BYTES
        )
        run(role + "-aiger", [str(tools.abc), "-c", _blif_script(role)], min(timeout, 60))
        for stem, name in steps.items():
            if stem.startswith(role + "-") and not name.endswith(".gz"):
                if time.monotonic() - began > cohort_seconds:
                    raise ValueError("The complete native checkpoint cohort is incomplete")
                compress_native_artifact(directory / name)

    if not streamed:
        try:
            for stem in flows:
                yosys(stem)
        except SynthesisToolError as error:
            if not error.timed_out or time.monotonic() - began > cohort_seconds - timeout:
                raise
            preserve_checkpoint_timeout(directory)
            flows, steps = _export_plan(True)
            write_flows()
            streamed = True
    if streamed:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reference = pool.submit(role_exports, "reference")
            mapped = pool.submit(role_exports, "mapped")
            reference.result()
            mapped.result()
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
