# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Function synthesis case and real mapped-statistics tests

"""Check the resource records against emitted RTL and real mapped circuits."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from sc_neurocore.compiler import emit_function_unit_verilog
from tools.function_synthesis_records import (
    SynthesisCase,
    declared_table_bits,
    read_synthesis_resources,
    synthesis_cases,
)
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import run_function_synthesis


class _NativeFixtureToolchain(SynthesisToolchain):
    """Preserve the original fixture's separate Yosys and ABC command ceilings."""

    __slots__ = ()

    def run(
        self,
        command: list[str],
        directory: Path,
        log: Path,
        *,
        timeout: float,
    ) -> None:
        """Run the actual inspected tool under its original fixture ceiling.

        Parameters
        ----------
        command : list of str
            Actual inspected native binary and its unchanged arguments.
        directory, log : pathlib.Path
            Real working directory and retained native output.
        timeout : float
            Caller deadline, capped at 120 seconds for Yosys and 45 for ABC.

        """
        ceiling = 45.0 if Path(command[0]).resolve() == self.abc else 120.0
        super().run(command, directory, log, timeout=min(timeout, ceiling))


def _map_report(root: Path, case: SynthesisCase) -> Path:
    """Map the real circuit through the public complete-domain proof pipeline."""
    tools = _NativeFixtureToolchain.inspect(root / "inspection")
    work = root / case.function
    measurement = run_function_synthesis(case, work, tools, synthesis_timeout=120, proof_timeout=90)
    proof = measurement.to_dict()["proof"]
    assert isinstance(proof, dict)
    assert proof == {
        "status": "equivalent",
        "method": measurement.proof_method,
        "input_bits": case.input_bits,
        "output_bits": case.data_width,
        "latches": 0,
    }
    return work / "stat.json"


@pytest.fixture(scope="module")
def wire_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Map the wire separately so negative tests do not repeat DSP synthesis per shard."""
    return _map_report(tmp_path_factory.mktemp("wire-statistics"), SynthesisCase("floor", 8, 0))


@pytest.fixture(scope="module")
def dsp_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Map the DSP-bearing circuit only in the test which consumes its counts."""
    return _map_report(tmp_path_factory.mktemp("dsp-statistics"), SynthesisCase("sin", 8, 4))


def test_complete_case_inventory_includes_unsupported_legacy_units() -> None:
    """Preserve every matrix disposition and each independently driven input."""
    cases = synthesis_cases()
    assert len(cases) == len({case.key for case in cases}) == 372
    assert sum(case.supported for case in cases) == 270
    assert sum(not case.supported for case in cases) == 102
    assert SynthesisCase("sin", 32, 16).key == "v2/sin/Q16.16"
    assert SynthesisCase("atan2", 32, 16).input_bits == 64
    assert SynthesisCase("sin", 32, 16).input_bits == 32
    with pytest.raises(ValueError, match="Unknown canonical function"):
        SynthesisCase("expit", 16, 8)
    with pytest.raises(ValueError, match="function_lowering"):
        SynthesisCase("sin", 16, 8, "unknown")
    with pytest.raises(TypeError, match="data_width"):
        SynthesisCase("sin", True, 8)
    with pytest.raises(ValueError, match="words of 2 to 32"):
        SynthesisCase("sin", 64, 8)


@pytest.mark.parametrize("case", synthesis_cases(), ids=lambda case: case.key)
def test_declared_table_bits_match_real_emitted_case_entries(case: SynthesisCase) -> None:
    """Count actual declared RTL entries without treating them as physical RAM."""
    if not case.supported:
        with pytest.raises(ValueError, match="No standalone implementation"):
            declared_table_bits(case)
        with pytest.raises(ValueError, match="has no"):
            emit_function_unit_verilog(
                case.function,
                data_width=case.data_width,
                fraction=case.fraction,
                function_lowering=case.lowering,
            )
        return
    rtl = emit_function_unit_verilog(
        case.function,
        data_width=case.data_width,
        fraction=case.fraction,
        function_lowering=case.lowering,
    )
    widths = re.findall(r"^\s+(?:\d+'d)?\d+:\s+\w+\s*=\s*-?(\d+)'sd\d+;", rtl, re.M)
    assert declared_table_bits(case) == sum(int(width) for width in widths)


def test_real_mapping_distinguishes_zero_cells_and_dsp_blocks(
    wire_report: Path,
    dsp_report: Path,
) -> None:
    """Read the actual mapped zero-cell wire and DSP-bearing polynomial circuit."""
    wire = read_synthesis_resources(wire_report)
    polynomial = read_synthesis_resources(dsp_report)
    assert wire.to_dict() == {
        "total_cells": 0,
        "cell_types": {},
        "dsp_blocks": 0,
        "remaining_memory_bits": 0,
    }
    assert polynomial.total_cells > 0
    assert polynomial.dsp_blocks > 0
    assert polynomial.remaining_memory_bits == 0
    assert polynomial.total_cells == sum(count for _, count in polynomial.cell_types)
    assert polynomial.to_dict()["dsp_blocks"] == polynomial.dsp_blocks


def test_reloaded_native_library_models_preserve_complete_physical_counts(
    dsp_report: Path, tmp_path: Path
) -> None:
    """Count actual primitive instances when native statistics separate their library models."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    shutil.copyfile(dsp_report.parent / "mapped.v", tmp_path / "mapped.v")
    (tmp_path / "replay.ys").write_text(
        "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
        "read_verilog mapped.v\nhierarchy -check -top function_unit\n"
        "tee -o stat.json stat -json -top function_unit\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "replay.ys"],
        tmp_path,
        tmp_path / "native-replay.log",
        timeout=90,
    )
    tools.check_library_log(tmp_path / "native-replay.log")
    observed = read_synthesis_resources(tmp_path / "stat.json")
    original = read_synthesis_resources(dsp_report)
    assert observed == original and observed.dsp_blocks == 18
    assert dict(observed.cell_types)["CARRY4"] > 0
    native = json.loads((tmp_path / "stat.json").read_text())["modules"]["\\function_unit"]
    assert observed.total_cells == native["num_cells"] + native.get("num_submodules", 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_processes", 1),
        ("num_processes", -1),
        ("num_processes", False),
        ("num_cells", 1),
        ("num_cells", -1),
        ("num_cells", True),
        ("num_cells", "0"),
        ("num_cells", 0.0),
        ("num_submodules", 1),
        ("num_submodules", -1),
        ("num_submodules", True),
        ("num_submodules", "0"),
        ("num_submodules", 0.0),
        ("num_cells_by_type", []),
        ("num_cells_by_type", {"LUT6": False}),
        ("num_cells_by_type", {"LUT6": -1}),
        ("num_memory_bits", None),
        ("num_memory_bits", True),
        ("num_memory_bits", -1),
        ("num_memory_bits", "0"),
    ],
)
def test_malformed_real_mapped_counts_are_refused(
    wire_report: Path, tmp_path: Path, field: str, value: object
) -> None:
    """Refuse corrupted statistics before they can become successful report rows."""
    payload: dict[str, object] = json.loads(wire_report.read_text())
    modules = payload["modules"]
    assert isinstance(modules, dict)
    module = modules["\\function_unit"]
    assert isinstance(module, dict)
    module[field] = value
    report = tmp_path / "corrupt-stat.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_synthesis_resources(report)


@pytest.mark.parametrize("boundary", ["report", "modules", "module", "duplicate", "nan"])
def test_corrupt_real_report_structure_is_refused(
    wire_report: Path, tmp_path: Path, boundary: str
) -> None:
    """Reject missing objects, duplicated counts and non-finite JSON values."""
    payload: dict[str, object] = json.loads(wire_report.read_text())
    if boundary == "modules":
        payload["modules"] = []
    elif boundary == "module":
        payload["modules"] = {"\\function_unit": []}
    elif boundary == "nan":
        payload["invalid_number"] = float("nan")
    text = json.dumps([] if boundary == "report" else payload)
    if boundary == "duplicate":
        assert '"num_cells": 0,' in text
        text = text.replace('"num_cells": 0,', '"num_cells": 0, "num_cells": 0,', 1)
    report = tmp_path / "corrupt-stat.json"
    report.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_synthesis_resources(report)
