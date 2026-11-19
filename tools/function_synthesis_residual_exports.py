# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded native residual graph exports

"""Export complete residual graphs with retained command and checkpoint provenance."""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tools.function_synthesis_aiger import AigerInterface, digest
from tools.function_synthesis_checkpoints import _MAX_CHECKPOINT_BYTES, _checkpoint, _command
from tools.function_synthesis_exports import (
    LARGE_LUT_EXPANSION,
    lut_gate_expansion,
    module_json_flow,
    validate_blif_output_drivers,
)
from tools.function_synthesis_json import (
    MAX_NETLIST_BYTES,
    native_netlist_artifacts,
    normalize_native_netlist,
)
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_toolchain import SynthesisToolchain

_TOP = "hierarchy -check -top function_unit\n"
_PREPARE = "proc\nmemory_map\nopt\nwreduce\nopt\n"
_BLIF_SCRIPT = "read_blif physical.blif; strash; write_aiger -s physical.aig; print_io"
_BLIF_NODE_THRESHOLD = 1 << 16
_PRE_LUT_SPLIT_CELLS = 500
_XILINX_MAP = "synth_xilinx -top function_unit -family xc7 -noiopad -noclkbuf -run "
_NATIVE_FRONTEND = re.compile(
    r"^[ .0-9]+Executing (?:Verilog-2005|SystemVerilog) frontend: (/.*)$", re.M
)


def _uses_variable_divisor(module: Netlist) -> bool:
    """Recognize native division whose denominator depends on input signals."""
    return any(
        cell.kind == "$div" and any(isinstance(bit, int) for bit in cell.connections["B"])
        for cell in module.cells.values()
    )


def _readback(checkpoint: str) -> str:
    """Reload a complete native checkpoint and require its original top module."""
    return f"read_rtlil {checkpoint}\n" + _TOP


def _compact_handoff(prefix: str, checkpoint: str) -> str:
    """Keep the complete checked top while shortening derived diagnostic metadata."""
    return (
        _compact_names(prefix)
        + f"select -module function_unit\nwrite_rtlil -selected {checkpoint}\n"
    )


def _compact_names(prefix: str) -> str:
    """Shorten private names and omit cell and wire source locations."""
    return (
        "tee -q attrmap -remove src function_unit\n"
        f"rename -enumerate -pattern _{prefix}_% function_unit/w:* function_unit/c:*\n"
    )


def _blif_command(directory: Path, tools: SynthesisToolchain) -> None:
    """Require the exact inspected ABC command and its sixty-second outer deadline."""
    receipt: object = json.loads((directory / "physical-strash.command.json").read_bytes())
    if not isinstance(receipt, list) or not all(isinstance(item, str) for item in receipt):
        raise ValueError("The retained BLIF conversion requires its original argument vector")
    argv = [item for item in receipt if isinstance(item, str)]
    if (
        len(argv) != 11
        or argv[:6]
        != [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
        ]
        or argv[7:] != [str(tools.abc), "-s", "-c", _BLIF_SCRIPT]
    ):
        raise ValueError("The retained BLIF conversion differs from the bounded native command")
    seconds = float(argv[6])
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("The retained BLIF conversion exceeds its native deadline")


def _check_mapping_library(log: Path, tools: SynthesisToolchain, model: str) -> None:
    """Require the phase's mapping models from the inspected library."""
    actual = {Path(path).resolve() for path in _NATIVE_FRONTEND.findall(log.read_text())}
    xilinx = tools.data_directory / "xilinx"
    required = {tools.data_directory / "techmap.v", xilinx / model}
    # Newer Yosys maps flip-flops before the remaining XC7 cells.
    allowed = required | {xilinx / "ff_map.v"} if model == "cells_map.v" else required
    if not required <= actual <= allowed:
        raise ValueError("The retained native division replay used a different technology model")


def _pre_lut_flows(
    module: Netlist,
) -> tuple[dict[str, str], dict[str, str]]:
    """Restore the original pre-LUT mapping through complete native phase boundaries."""
    prefix = "read_verilog -sv unit.v\n" + _TOP + "proc\nopt_clean\n"
    flows: dict[str, str] = {}
    checkpoints: dict[str, str] = {}
    if _uses_variable_divisor(module) or len(module.cells) >= _PRE_LUT_SPLIT_CELLS:
        flows["pre-lut-begin"] = (
            prefix + _XILINX_MAP + "begin:coarse\nwrite_rtlil pre-lut.begin.il.gz\n"
        )
        checkpoints["pre-lut-begin"] = "pre-lut.begin.il.gz"
        for stem, phase, before, after in (
            ("pre-lut-coarse", "coarse:fine", "begin", "coarse"),
            ("pre-lut-fine", "fine:map_cells", "coarse", "fine"),
        ):
            checkpoints[stem] = f"pre-lut.{after}.il.gz"
            flows[stem] = (
                _readback(f"pre-lut.{before}.il.gz")
                + _XILINX_MAP
                + f"{phase}\nwrite_rtlil {checkpoints[stem]}\n"
            )
        prefix = _readback("pre-lut.fine.il.gz")
        mapping = "map_cells:map_luts"
    else:
        mapping = "begin:map_luts"
    flows["pre-lut-mapping"] = prefix + _XILINX_MAP + mapping + "\nwrite_rtlil pre-lut.il.gz\n"
    checkpoints["pre-lut-mapping"] = "pre-lut.il.gz"
    flows["pre-lut"] = (
        _readback("pre-lut.il.gz")
        + "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v function_unit/t:CARRY4\n"
        + "rename -enumerate function_unit/w:* function_unit/c:* %u function_unit/t:DSP48E1 %d\n"
        + module_json_flow("pre-lut.json")
    )
    return flows, checkpoints


def _check_pre_lut_library(stem: str, log: Path, tools: SynthesisToolchain, split: bool) -> None:
    """Bind every native pre-LUT stage to the inspected XC7 library."""
    if stem == "pre-lut-coarse":
        actual = {Path(path).resolve() for path in _NATIVE_FRONTEND.findall(log.read_text())}
        if not actual or any(not path.is_relative_to(tools.data_directory) for path in actual):
            raise ValueError("The pre-LUT coarse stage used a different technology library")
    elif stem == "pre-lut-fine":
        _check_mapping_library(log, tools, "arith_map.v")
    elif split and stem == "pre-lut-mapping":
        _check_mapping_library(log, tools, "cells_map.v")
    else:
        tools.check_library_log(log)


def verify_pre_lut_exports(module: Netlist, directory: Path, tools: SynthesisToolchain) -> None:
    """Validate retained pre-LUT commands, native serialization and complete checkpoints."""
    normalized = directory / "pre-lut-native"
    if normalized.exists() or normalized.is_symlink():
        native_netlist_artifacts(normalized)
        if digest(normalized / "mapped.json") != digest(directory / "pre-lut.json"):
            raise ValueError(
                "The pre-LUT checkpoint differs from its lossless native serialization"
            )
    flows, checkpoints = _pre_lut_flows(module)
    for stem, expected in flows.items():
        script = directory / (stem + ".ys")
        log = directory / (stem + ".log")
        receipt = directory / (stem + ".command.json")
        if any(path.is_symlink() or not path.is_file() for path in (script, log, receipt)):
            raise ValueError("A pre-LUT proof requires regular retained checkpoint artifacts")
        if script.read_text(encoding="utf-8") != expected:
            raise ValueError("A retained original checkpoint preparation flow changed")
        _command(directory, stem, tools)
        _check_pre_lut_library(stem, log, tools, "pre-lut-begin" in flows)
        if stem in checkpoints:
            _checkpoint(directory / checkpoints[stem])


def export_pre_lut(
    module: Netlist, directory: Path, tools: SynthesisToolchain, timeout: float
) -> None:
    """Acquire a complete native pre-LUT checkpoint under the same case deadline."""
    flows, checkpoints = _pre_lut_flows(module)
    for stem, script in flows.items():
        path = directory / (stem + ".ys")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(script)
        log = directory / (stem + ".log")
        tools.run([str(tools.yosys), "-Q", "-T", "-s", path.name], directory, log, timeout=timeout)
        _check_pre_lut_library(stem, log, tools, "pre-lut-begin" in flows)
        if stem in checkpoints:
            _checkpoint(directory / checkpoints[stem])
    checkpoint = directory / "pre-lut.json"
    if checkpoint.stat().st_size > MAX_NETLIST_BYTES:
        normalized = directory / "pre-lut-native"
        normalized.mkdir()
        checkpoint.rename(normalized / "mapped.json")
        normalize_native_netlist(normalized)
        os.link(normalized / "mapped.json", checkpoint)


def _word_flow(division_replay: bool) -> tuple[dict[str, str], dict[str, str]]:
    """Replay native arithmetic mapping when the divisor depends on signals."""
    scripts: dict[str, str] = {}
    checkpoints: dict[str, str] = {}
    prefix = "read_json word-cut.json\n" + _TOP
    lowering = "abc -g AND -script word-abc.script"
    if division_replay:
        scripts.update(
            {
                "word-native-mapping": (
                    "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
                    "read_verilog -lib +/xilinx/cells_xtra.v\n"
                    + prefix
                    + _XILINX_MAP
                    + "coarse:fine\nwrite_rtlil word.coarse.il.gz\n"
                ),
                "word-native-fine": _readback("word.coarse.il.gz")
                + _XILINX_MAP
                + "fine:map_cells\nwrite_rtlil word.fine.il.gz\n",
                "word-native-cells": _readback("word.fine.il.gz")
                + _XILINX_MAP
                + "map_cells:map_luts\nwrite_rtlil word.pre-lut.il.gz\n",
                "word-native-carry": _readback("word.pre-lut.il.gz")
                + "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v function_unit/t:CARRY4\n"
                + "rename -enumerate function_unit/w:* function_unit/c:* %u function_unit/t:DSP48E1 %d\n"
                + "write_rtlil word.primitives.il.gz\n",
            }
        )
        checkpoints.update(
            {
                "word-native-mapping": "word.coarse.il.gz",
                "word-native-fine": "word.fine.il.gz",
                "word-native-cells": "word.pre-lut.il.gz",
                "word-native-carry": "word.primitives.il.gz",
            }
        )
        prefix = _readback("word.primitives.il.gz")
        lowering = "aigmap"
    scripts["word-prepare"] = prefix + _PREPARE + "write_rtlil word.prepared.il.gz\n"
    scripts["word-map"] = (
        _readback("word.prepared.il.gz") + "techmap\nopt_clean\nwrite_rtlil word.il.gz\n"
    )
    scripts["word"] = (
        _readback("word.il.gz")
        + lowering
        + "\nopt_clean\ncheck -assert\nwrite_aiger -symbols word.aig\n"
    )
    checkpoints.update({"word-prepare": "word.prepared.il.gz", "word-map": "word.il.gz"})
    return scripts, checkpoints


@dataclass(frozen=True, slots=True)
class ResidualExports:
    """Describe ordered native commands and all retained residual export inputs."""

    scripts: dict[str, str]
    checkpoints: dict[str, str]
    blif: bool
    division_replay: bool

    def run(
        self, directory: Path, tools: SynthesisToolchain, timeout: float, interface: AigerInterface
    ) -> None:
        """Export independently checked graphs in parallel under one shared cohort."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            word = pool.submit(self._run_role, "word", directory, tools, timeout, interface)
            physical = pool.submit(self._run_role, "physical", directory, tools, timeout, interface)
            word.result()
            physical.result()

    def _run_role(
        self,
        role: str,
        directory: Path,
        tools: SynthesisToolchain,
        timeout: float,
        interface: AigerInterface,
    ) -> None:
        """Keep the stages within one role ordered and retain their separate receipts."""
        for stem in self.scripts:
            if stem.split("-", 1)[0] != role:
                continue
            tools.run(
                [str(tools.yosys), "-Q", "-T", "-s", stem + ".ys"],
                directory,
                directory / (stem + ".log"),
                timeout=timeout,
            )
            if stem in self.checkpoints:
                _checkpoint(directory / self.checkpoints[stem])
        if role == "physical" and self.blif:
            validate_blif_output_drivers(
                directory / "physical.blif", interface, _MAX_CHECKPOINT_BYTES
            )
            tools.run(
                [str(tools.abc), "-c", _BLIF_SCRIPT],
                directory,
                directory / "physical-strash.log",
                timeout=min(timeout, 60.0),
            )

    def record(
        self, directory: Path, tools: SynthesisToolchain, interface: AigerInterface
    ) -> dict[str, object]:
        """Recompute the complete export record without running a tool or changing a file."""
        names = {
            stem + extension
            for stem in self.scripts
            for extension in (".ys", ".log", ".command.json")
        }
        for role in ("word", "physical"):
            names.update((role + "-cut.json", role + ".aig"))
        if not self.division_replay:
            names.add("word-abc.script")
        if self.blif:
            names.update(("physical.blif", "physical-strash.log", "physical-strash.command.json"))
        checkpoints = set(self.checkpoints.values())
        for name in names | checkpoints:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("A retained residual export requires regular artifacts")
            if not name.endswith((".aig", ".il.gz", ".blif", "-cut.json")) and (
                path.stat().st_size > MAX_NETLIST_BYTES
            ):
                raise ValueError("A retained residual export requires bounded text artifacts")
        for stem in self.scripts:
            _command(directory, stem, tools)
        if self.blif:
            validate_blif_output_drivers(
                directory / "physical.blif", interface, _MAX_CHECKPOINT_BYTES
            )
            _blif_command(directory, tools)
        tools.check_library_log(directory / "physical-primitives.log")
        if self.division_replay:
            tools.check_library_log(directory / "word-native-mapping.log")
            tools.check_library_log(directory / "word-native-carry.log")
            _check_mapping_library(directory / "word-native-fine.log", tools, "arith_map.v")
            _check_mapping_library(directory / "word-native-cells.log", tools, "cells_map.v")
        return {
            "schema_version": (
                "sc-neurocore.primitive-residual-exports.v3"
                if self.blif or self.division_replay
                else "sc-neurocore.primitive-residual-exports.v2"
            ),
            "toolchain_sha256": tools.identity,
            "artifact_sha256": {name: digest(directory / name) for name in sorted(names)},
            "checkpoints": {name: _checkpoint(directory / name) for name in sorted(checkpoints)},
        }


def plan_residual_exports(word: Netlist, physical: Netlist) -> ResidualExports:
    """Derive bounded complete exports from both actual synthesis checkpoints."""
    division_replay = _uses_variable_divisor(word)
    expansion = lut_gate_expansion(physical)
    blif = expansion >= _BLIF_NODE_THRESHOLD
    scripts, checkpoints = _word_flow(division_replay)
    scripts["physical-primitives"] = (
        "read_verilog -lib -specify +/xilinx/cells_sim.v\nread_json physical-cut.json\n"
        + _TOP
        + "techmap -wb -D EQUIV -autoproc -map +/xilinx/cells_sim.v\n"
        + "write_rtlil physical.primitives.il.gz\n"
    )
    checkpoints["physical-primitives"] = "physical.primitives.il.gz"
    if blif:
        split_fold = expansion >= LARGE_LUT_EXPANSION
        compact = _compact_names("expanded") if split_fold else ""
        scripts["physical-lower"] = (
            _readback("physical.primitives.il.gz")
            + "proc\nmemory_map\n"
            + compact
            + "write_rtlil physical.lowered.il.gz\n"
        )
        checkpoints["physical-lower"] = "physical.lowered.il.gz"
        fold = _readback("physical.lowered.il.gz") + "opt_expr\nopt_clean\n"
        if split_fold:
            scripts["physical-fold"] = fold + "write_rtlil physical.folded.il.gz\n"
            checkpoints["physical-fold"] = "physical.folded.il.gz"
            scripts["physical-width"] = (
                _readback("physical.folded.il.gz") + "wreduce\nwrite_rtlil physical.reduced.il.gz\n"
            )
            checkpoints["physical-width"] = "physical.reduced.il.gz"
        else:
            scripts["physical-fold"] = fold + "wreduce\nwrite_rtlil physical.reduced.il.gz\n"
            checkpoints["physical-fold"] = "physical.reduced.il.gz"
        scripts.update(
            {
                "physical-clean": _readback("physical.reduced.il.gz")
                + "opt_expr\nopt_clean\n"
                + _compact_handoff("logic", "physical.prepared.il.gz"),
                "physical-map": _readback("physical.prepared.il.gz")
                + "techmap\nopt_clean\n"
                + _compact_handoff("gate", "physical.il.gz"),
                "physical-check": _readback("physical.il.gz") + "check -assert function_unit\n",
                "physical": _readback("physical.il.gz")
                + "write_blif -noalias -top function_unit physical.blif\n",
            }
        )
        checkpoints["physical-clean"] = "physical.prepared.il.gz"
    else:
        scripts["physical-prepare"] = (
            _readback("physical.primitives.il.gz")
            + _PREPARE
            + "write_rtlil physical.prepared.il.gz\n"
        )
        checkpoints["physical-prepare"] = "physical.prepared.il.gz"
        scripts["physical-map"] = (
            _readback("physical.prepared.il.gz")
            + "techmap\nopt_clean\nwrite_rtlil physical.il.gz\n"
        )
        scripts["physical"] = _readback("physical.il.gz") + (
            "aigmap\nopt_clean\ncheck -assert\nwrite_aiger -symbols physical.aig\n"
        )
    checkpoints["physical-map"] = "physical.il.gz"
    return ResidualExports(scripts, checkpoints, blif, division_replay)
