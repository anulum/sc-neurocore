# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-bound adjacent mapping integration

"""Exercise complete mapping stages through the real synthesis producer and reader."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from tools.function_synthesis_dsp_proof import read_primitive_residual
from tools.function_synthesis_json import MAX_NETLIST_BYTES
from tools.function_synthesis_mapping_stages import read_mapping_stages
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    read_function_synthesis_measurement,
    run_function_synthesis,
)


@pytest.fixture(scope="module")
def mapping_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SynthesisCase, Path, SynthesisToolchain]:
    """Retain both native mapping proofs acquired by the actual public sine producer."""
    root = tmp_path_factory.mktemp("actual-adjacent-mappings")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("sin", 8, 4)
    run_function_synthesis(case, root / "measurement", tools)
    return case, root / "measurement", tools


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Observe every retained byte identity and timestamp across a real reader call."""
    return {
        p.relative_to(root).as_posix(): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in root.rglob("*")
        if p.is_file()
    }


def test_public_producer_and_reader_bind_both_complete_mapping_stages(
    mapping_case: tuple[SynthesisCase, Path, SynthesisToolchain],
) -> None:
    """Preserve the identical intermediate and all original bits in both native proofs."""
    case, root, tools = mapping_case
    directory = root / "composition-proof"
    before = _snapshot(root)
    stages = read_mapping_stages(
        Netlist.read(directory / "post-dsp.json"),
        Netlist.read(directory / "pre-lut.json"),
        Netlist.read(root / "mapped.json"),
        {"argument0": case.data_width},
        case.data_width,
        tools,
        directory,
    )
    assert set(stages) == {"pre_mapping_primitives", "primitives"}
    for key, name in (
        ("pre_mapping_primitives", "word-mapping-proof"),
        ("primitives", "primitive-proof"),
    ):
        actual = json.loads((directory / name / "primitive-residual.json").read_text())
        assert stages[key] == actual
        assert actual["original_input_bits"] == actual["original_output_bits"] == 8
        native = json.loads((directory / name / "gia-proof.json").read_text())
        assert native["input_bits"] >= 8 and native["all_compared_output_bits"] >= 8
        assert native["status"] == "equivalent"
    assert read_function_synthesis_measurement(case, root, tools).proof_method == "compositional"
    assert _snapshot(root) == before


def test_changed_shared_intermediate_cannot_reuse_either_native_component(
    mapping_case: tuple[SynthesisCase, Path, SynthesisToolchain], tmp_path: Path
) -> None:
    """Refuse a changed actual checkpoint rather than trusting unchanged native proof labels."""
    case, root, tools = mapping_case
    directory = root / "composition-proof"
    payload = json.loads((directory / "pre-lut.json").read_text())
    module = payload["modules"]["function_unit"]
    old = module["ports"]["result"]["bits"][0]
    all_bits = [
        bit
        for cell in module["cells"].values()
        for values in cell["connections"].values()
        for bit in values
    ]
    all_bits.extend(bit for port in module["ports"].values() for bit in port["bits"])
    all_bits.extend(bit for net in module["netnames"].values() for bit in net["bits"])
    fresh = max(bit for bit in all_bits if isinstance(bit, int)) + 1
    module["ports"]["result"]["bits"][0] = fresh
    module["netnames"]["result"]["bits"][0] = fresh
    module["cells"]["actual_result_inversion"] = {
        "hide_name": 0,
        "type": "$_NOT_",
        "parameters": {},
        "attributes": {},
        "port_directions": {"A": "input", "Y": "output"},
        "connections": {"A": [old], "Y": [fresh]},
    }
    changed = tmp_path / "changed-intermediate.json"
    changed.write_text(json.dumps(payload))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="cut or export flow differs"):
        read_mapping_stages(
            Netlist.read(directory / "post-dsp.json"),
            Netlist.read(changed),
            Netlist.read(root / "mapped.json"),
            {"argument0": case.data_width},
            case.data_width,
            tools,
            directory,
        )
    assert _snapshot(root) == before


def test_generic_cut_function_change_is_refused_before_native_verdict_readback(
    mapping_case: tuple[SynthesisCase, Path, SynthesisToolchain], tmp_path: Path
) -> None:
    """Bind actual retained generic output edges even when native result files remain valid."""
    case, root, tools = mapping_case
    directory = root / "composition-proof"
    copied = tmp_path / "changed-cut"
    shutil.copytree(directory / "primitive-proof", copied)
    path = copied / "word-cut.json"
    payload = json.loads(path.read_text())
    module = payload["modules"]["function_unit"]
    module["ports"]["result"]["bits"][0] = "0"
    module["netnames"]["result"]["bits"][0] = "0"
    path.write_text(json.dumps(payload))
    before = _snapshot(copied)
    with pytest.raises(ValueError, match="cut or export flow differs"):
        read_primitive_residual(
            Netlist.read(directory / "pre-lut.json"),
            Netlist.read(root / "mapped.json"),
            {"argument0": case.data_width},
            case.data_width,
            tools,
            copied,
        )
    assert _snapshot(copied) == before


@pytest.mark.parametrize("stage", ["word-mapping-proof", "primitive-proof"])
@pytest.mark.parametrize(
    "damage", ["directory-link", "missing-cut", "cut-link", "cut-directory", "oversized-cut"]
)
def test_retained_component_requires_bounded_regular_evidence(
    mapping_case: tuple[SynthesisCase, Path, SynthesisToolchain],
    tmp_path: Path,
    stage: str,
    damage: str,
) -> None:
    """Refuse redirected, absent, irregular or oversized evidence in either real component."""
    case, root, tools = mapping_case
    composition = root / "composition-proof"
    copied = tmp_path / "component"
    shutil.copytree(composition / stage, copied, copy_function=os.link)
    selected = copied
    path = copied / "word-cut.json"
    if damage == "directory-link":
        selected = tmp_path / "component-link"
        selected.symlink_to(copied, target_is_directory=True)
    else:
        path.unlink()
        if damage == "cut-link":
            path.symlink_to(composition / stage / "word-cut.json")
        elif damage == "cut-directory":
            path.mkdir()
        elif damage == "oversized-cut":
            with path.open("wb") as stream:
                stream.truncate(MAX_NETLIST_BYTES + 1)
    original = _snapshot(root)
    before = {
        p.name: (p.lstat().st_mode, p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in copied.iterdir()
    }
    source, target = (
        (composition / "post-dsp.json", composition / "pre-lut.json")
        if stage == "word-mapping-proof"
        else (composition / "pre-lut.json", root / "mapped.json")
    )
    with pytest.raises(ValueError, match="directory link|bounded regular artifacts"):
        read_primitive_residual(
            Netlist.read(source),
            Netlist.read(target),
            {"argument0": case.data_width},
            case.data_width,
            tools,
            selected,
        )
    assert _snapshot(root) == original
    assert before == {
        p.name: (p.lstat().st_mode, p.lstat().st_size, p.lstat().st_mtime_ns)
        for p in copied.iterdir()
    }
