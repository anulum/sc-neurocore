# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete original-domain compositional proof contracts

"""Exercise actual integrated DSP proofs and refusal of damaged retained evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from tools.function_synthesis_cache import FunctionSynthesisCache
from tools.function_synthesis_dsp_proof import prove_primitive_residual
from tools.function_synthesis_netlist import Netlist
from tools.function_synthesis_proof import (
    prove_compositional_mapping,
    read_compositional_mapping_proof,
)
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_source_proof import read_source_normalization
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    FunctionSynthesisMeasurement,
    check_mapped_equivalence,
    read_function_synthesis_measurement,
    run_function_synthesis,
)

ProofCase = tuple[SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement]


def _snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    """Read every actual artifact identity and original modification time."""
    return {
        path.relative_to(directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module", params=[8, 32])
def proof_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ProofCase:
    """Map and prove both complete original domains through the production runner."""
    width = int(request.param)
    root = tmp_path_factory.mktemp(f"actual-composition-{width}")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("sin", width, width // 2)
    measurement = run_function_synthesis(case, root / "measurement", tools)
    return case, root / "measurement", tools, measurement


def test_complete_original_domain_and_physical_resources(proof_case: ProofCase) -> None:
    """Preserve physical DSPs, full primary ports and explicit compositional proof method."""
    case, root, tools, measurement = proof_case
    assert measurement.resources.dsp_blocks == (18 if case.data_width == 8 else 22)
    physical = Netlist.read(root / "mapped.json")
    assert measurement.resources.total_cells == len(physical.cells)
    assert dict(measurement.resources.cell_types) == Counter(
        cell.kind for cell in physical.cells.values()
    )
    assert measurement.to_dict()["proof"] == {
        "status": "equivalent",
        "method": "compositional",
        "input_bits": case.input_bits,
        "output_bits": case.data_width,
        "latches": 0,
    }
    record = json.loads((root / "composition-proof/proof-record.json").read_text())
    composition = root / "composition-proof"
    structure = json.loads(
        (composition / "source-structure/structural-comparison.json").read_text()
    )
    identical = structure["all_output_function_keys_identical"]
    assert isinstance(identical, bool)
    assert record["source_method"] == {True: "structural", False: "complete-aiger"}[identical]
    assert read_source_normalization(case, composition, tools) == record["source_method"]
    assert record["word"]["products_proven"] == 4
    assert record["word"]["unchanged_functions_checked"] == 36
    assert read_function_synthesis_measurement(case, root, tools) == measurement


def test_retained_composition_never_restamps_native_evidence(proof_case: ProofCase) -> None:
    """Repeated public reads preserve every source, proof, command and timestamp."""
    case, root, tools, measurement = proof_case
    before = _snapshot(root)
    proof = read_compositional_mapping_proof(case, root, tools)
    assert dict(proof.artifact_sha256) == dict(measurement.artifact_sha256)
    assert read_function_synthesis_measurement(case, root, tools) == measurement
    assert _snapshot(root) == before


def test_final_source_validation_charges_real_completed_compositional_proof(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Refuse mapped success when final source readback exhausts its actual shared allowance."""
    from dataclasses import replace

    case, original, tools, measurement = proof_case
    assert measurement.proof_method == "compositional"
    control, root = tmp_path / "positive", tmp_path / "negative"
    for destination in (control, root):
        destination.mkdir()
        for source in original.iterdir():
            if source.is_file():
                shutil.copyfile(source, destination / source.name)
    started = time.monotonic()
    check_mapped_equivalence(case, control, tools)
    elapsed = time.monotonic() - started
    timeout = min(90.0, max(5.0, elapsed * 2))
    allowance = min(3300.0, max(timeout + 5.0, elapsed * 4 + timeout))
    deadline = time.monotonic() + allowance
    bounded = replace(tools, cohort_deadline=deadline)
    delayed = False

    def delay_final_original_read(event: str, args: tuple[object, ...]) -> None:
        """Wait on actual source I/O only after the full native component record exists."""
        nonlocal delayed
        if (
            event == "open"
            and args[0] in (root / "unit.v", str(root / "unit.v"))
            and (root / "composition-proof/proof-record.json").is_file()
            and not delayed
        ):
            delayed = True
            time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)

    delay_final_original_read.__dict__["__cantrace__"] = True
    sys.addaudithook(delay_final_original_read)
    with pytest.raises(
        ValueError, match="Mapped proof validation exhausted the shared cohort budget"
    ):
        check_mapped_equivalence(case, root, bounded, timeout=timeout, cohort_seconds=allowance)
    assert delayed
    assert read_compositional_mapping_proof(case, root, tools).artifact_sha256
    assert (root / "unit.v").read_bytes() == (control / "unit.v").read_bytes()


@pytest.mark.parametrize(
    "name",
    [
        "unit.v",
        "flow.ys",
        "mapped.json",
        "composition-proof/proof-record.json",
        "composition-proof/unit.v",
        "composition-proof/pre-dsp.ys",
        "composition-proof/post-dsp.ys",
        "composition-proof/original-words.ys",
        "composition-proof/pre-dsp.json",
        "composition-proof/post-dsp.json",
        "composition-proof/source-structure/structural-comparison.json",
        "composition-proof/word-proof/product-000.json",
        "composition-proof/primitive-proof/equivalence.log",
        "composition-proof/primitive-proof/word.ys",
        "composition-proof/primitive-proof/word-abc.script",
        "composition-proof/primitive-proof/word-cut.json",
        "composition-proof/contracts/contract-00/contract-record.json",
    ],
)
def test_changed_retained_artifact_is_preserved_and_refused(
    proof_case: ProofCase, tmp_path: Path, name: str
) -> None:
    """Refuse actual modified acquisition evidence without rebuilding or erasing it."""
    case, original, tools, _ = proof_case
    root = tmp_path / "damaged"
    shutil.copytree(original, root)
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    before = _snapshot(root)
    with pytest.raises(ValueError):
        read_function_synthesis_measurement(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize("name", ["composition-proof", "composition-proof/proof-record.json"])
def test_linked_retained_composition_is_refused(
    proof_case: ProofCase, tmp_path: Path, name: str
) -> None:
    """Refuse a linked proof directory or record even with genuine successful target bytes."""
    case, original, tools, _ = proof_case
    root = tmp_path / "linked"
    shutil.copytree(original, root)
    path = root / name
    target = tmp_path / "actual-retained-target"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="link"):
        read_compositional_mapping_proof(case, root, tools)
    assert path.is_symlink() and target.exists()


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_composition_deadline_preserves_success(
    proof_case: ProofCase, timeout: float
) -> None:
    """Validate the caller budget before opening or replacing any actual acquisition."""
    case, root, tools, _ = proof_case
    before = _snapshot(root)
    with pytest.raises(ValueError, match="finite positive"):
        prove_compositional_mapping(case, root, tools, timeout=timeout)
    assert _snapshot(root) == before


def test_incorrect_function_cannot_relabel_a_complete_proof(proof_case: ProofCase) -> None:
    """Keep source custody when a caller supplies another supported original function."""
    case, root, tools, _ = proof_case
    with pytest.raises(ValueError, match="different|differs"):
        read_compositional_mapping_proof(
            SynthesisCase("cos", case.data_width, case.fraction), root, tools
        )


def test_actual_compositional_cache_reuses_original_acquisition(
    tmp_path: Path,
) -> None:
    """Admit a real DSP cache transaction and preserve its original successful proof on reuse."""
    tools = SynthesisToolchain.inspect(tmp_path / "inspection")
    cache = FunctionSynthesisCache(tmp_path / "cache", tools)
    case = SynthesisCase("sin", 8, 4)
    assert cache.get(case) is None
    original = cache.measure(case)
    assert not original.reused and original.measurement.proof_method == "compositional"
    before = _snapshot(original.directory)
    readback = cache.get(case)
    repeated = cache.measure(case)
    assert readback is not None and readback.reused and repeated.reused
    assert readback.measurement == original.measurement == repeated.measurement
    assert original.recorded_at == readback.recorded_at == repeated.recorded_at
    assert _snapshot(original.directory) == before


def test_actual_physical_result_fault_cannot_pass_compositional_proof(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Refuse a connected result fault in the complete actual cost-bearing physical JSON."""
    case, original, tools, _ = proof_case
    root = tmp_path / "fault"
    root.mkdir()
    for path in original.iterdir():
        if path.is_file() and not path.name.startswith(("reference-io.", "mapped-io.")):
            shutil.copyfile(path, root / path.name)
    document = json.loads((root / "mapped.json").read_text())
    module = document["modules"]["function_unit"]
    old = module["ports"]["result"]["bits"][0]
    highest = max(
        bit
        for cell in module["cells"].values()
        for bits in cell["connections"].values()
        for bit in bits
        if isinstance(bit, int)
    )
    fresh = highest + 1
    module["ports"]["result"]["bits"][0] = fresh
    module["netnames"]["result"]["bits"][0] = fresh
    module["cells"]["connected_output_fault"] = {
        "type": "INV",
        "parameters": {},
        "port_directions": {"I": "input", "O": "output"},
        "connections": {"I": [old], "O": [fresh]},
    }
    (root / "mapped.json").write_text(json.dumps({"modules": {"function_unit": module}}))
    (root / "fault-export.ys").write_text(
        "read_verilog -lib -specify +/xilinx/cells_sim.v\n"
        "read_json mapped.json\nhierarchy -check -top function_unit\n"
        "write_verilog -noattr mapped.v\n"
        "tee -o stat.json stat -json -top function_unit\n"
    )
    tools.run(
        [str(tools.yosys), "-Q", "-T", "fault-export.ys"],
        root,
        root / "fault-export.log",
        timeout=90,
    )
    with pytest.raises(ValueError, match="Complete native GIA equivalence was not proved"):
        prove_compositional_mapping(case, root, tools, timeout=90)
    assert (
        "NOT EQUIVALENT" in (root / "composition-proof/primitive-proof/equivalence.log").read_text()
    )
    assert not (root / "composition-proof/primitive-proof/gia-proof.json").exists()


def test_non_generic_retained_word_cell_is_refused(proof_case: ProofCase, tmp_path: Path) -> None:
    """Reject a physical-library cell substituted into an already proved generic word cut."""
    case, original, tools, _ = proof_case
    root = tmp_path / "non-generic-retained-word"
    shutil.copytree(original, root)
    path = root / "composition-proof/primitive-proof/word-cut.json"
    document = json.loads(path.read_text())
    cell = next(iter(document["modules"]["function_unit"]["cells"].values()))
    assert cell["type"].startswith("$")
    cell["type"] = "INV"
    path.write_text(json.dumps(document))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="generic word export flow or actual cell domain"):
        read_compositional_mapping_proof(case, root, tools)
    assert _snapshot(root) == before


def test_non_generic_original_word_cell_is_refused_before_native_export(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Refuse an actual checkpoint with an unqualified library cell before using generic export."""
    case, original, tools, _ = proof_case
    document = json.loads((original / "composition-proof/post-dsp.json").read_text())
    cell = next(
        cell
        for cell in document["modules"]["function_unit"]["cells"].values()
        if cell["type"].startswith("$")
    )
    cell["type"] = "INV"
    path = tmp_path / "non-generic-checkpoint.json"
    path.write_text(json.dumps(document))
    proof = tmp_path / "refused-native-export"
    with pytest.raises(ValueError, match="non-generic actual cell"):
        prove_primitive_residual(
            Netlist.read(path),
            Netlist.read(original / "mapped.json"),
            {"argument0": case.data_width},
            case.data_width,
            tools,
            proof,
            timeout=90,
        )
    assert not (proof / "word-synthesis.command.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "root-flow",
        "root-link",
        "checkpoint-source",
        "checkpoint-flow",
        "structure-link",
        "structure-type",
        "structure-roots",
        "structure-incomplete",
        "word-count",
        "word-order",
        "product-cell",
        "product-width",
        "product-residue",
        "primitive-values",
        "primitive-count",
        "primitive-inputs",
        "primitive-outputs",
        "primitive-connected-inputs",
        "primitive-configuration",
        "word-flow",
        "word-abc",
        "unbound-link",
        "irregular-artifact",
    ],
)
def test_incomplete_retained_proof_components_are_refused_without_reacquisition(
    proof_case: ProofCase, tmp_path: Path, damage: str
) -> None:
    """Require complete original roots, products, primitive geometry and regular source closure."""
    case, original, tools, _ = proof_case
    root = tmp_path / "damaged-component"
    shutil.copytree(original, root)
    composition = root / "composition-proof"
    if damage in {"root-flow", "checkpoint-source", "checkpoint-flow", "word-flow", "word-abc"}:
        file_paths = {
            "root-flow": root / "flow.ys",
            "checkpoint-source": composition / "unit.v",
            "checkpoint-flow": composition / "pre-dsp.ys",
            "word-flow": composition / "primitive-proof/word.ys",
            "word-abc": composition / "primitive-proof/word-abc.script",
        }
        path = file_paths[damage]
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage in {"root-link", "structure-link"}:
        path = (
            root / "mapped.v"
            if damage == "root-link"
            else composition / "source-structure/structural-comparison.json"
        )
        target = tmp_path / "actual-component-target"
        path.rename(target)
        path.symlink_to(target)
    elif damage == "unbound-link":
        (composition / "extra-linked-artifact").symlink_to(original / "unit.v")
    elif damage == "irregular-artifact":
        os.mkfifo(composition / "extra-irregular-artifact")
    else:
        if damage.startswith("structure-"):
            path = composition / "source-structure/structural-comparison.json"
            value = json.loads(path.read_text())
            if damage == "structure-type":
                value = []
            elif damage == "structure-roots":
                value["original_output_roots"].pop()
            else:
                value["all_output_function_keys_identical"] = None
        elif damage in {"word-count", "word-order"}:
            path = composition / "word-proof/word-composition.json"
            value = json.loads(path.read_text())
            if damage == "word-count":
                value["products_proven"] += 1
            else:
                value["original_acyclic_order"].reverse()
        elif damage.startswith("product-"):
            path = composition / "word-proof/product-000.json"
            value = json.loads(path.read_text())
            if damage == "product-cell":
                value["original_cell"] += "_not_the_original_cell"
            elif damage == "product-width":
                value["complete_product_bits"] -= 1
            else:
                value["difference"] = [1]
        elif damage == "primitive-configuration":
            path = composition / "post-dsp.json"
            value = json.loads(path.read_text())
            cell = next(
                cell
                for cell in value["modules"]["function_unit"]["cells"].values()
                if cell["type"] == "DSP48E1"
            )
            cell["parameters"]["MREG"] = "1"
        else:
            path = composition / "primitive-proof/primitive-residual.json"
            value = json.loads(path.read_text())
            fields = {
                "primitive-values": "universal_primitive_values",
                "primitive-count": "physical_dsp_count",
                "primitive-inputs": "original_input_bits",
                "primitive-outputs": "original_output_bits",
                "primitive-connected-inputs": "compared_primitive_input_bits",
            }
            name = fields[damage]
            value[name] = True if damage == "primitive-values" else value[name] - 1
        path.write_text(json.dumps(value))
    before = _snapshot(root) if damage != "irregular-artifact" else None
    with pytest.raises(ValueError):
        read_compositional_mapping_proof(case, root, tools)
    if before is not None:
        assert _snapshot(root) == before
    else:
        assert (composition / "extra-irregular-artifact").is_fifo()


@pytest.mark.parametrize("proof_case", [8], indirect=True)
def test_acquisition_change_during_native_composition_cannot_be_accepted(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Refuse a real mapping-artifact change at the native residual subprocess boundary."""
    case, original, tools, _ = proof_case
    root = tmp_path / "concurrently-changed-acquisition"
    root.mkdir()
    for path in original.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    changed = threading.Event()
    primitive = root / "composition-proof/primitive-proof"

    def change_acquisition(event: str, args: tuple[object, ...]) -> None:
        """Modify the actual retained mapping output during the complete native residual proof."""
        if (
            event == "subprocess.Popen"
            and args[2] in (primitive, str(primitive), str(primitive).encode())
            and (primitive / "physical-synthesis.command.json").is_file()
            and not changed.is_set()
        ):
            with (root / "mapped.v").open("ab") as stream:
                stream.write(b"\n// actual acquisition changed at its native subprocess boundary\n")
            changed.set()

    change_acquisition.__dict__["__cantrace__"] = True
    sys.addaudithook(change_acquisition)
    with pytest.raises(ValueError, match="acquisition or proof implementation changed"):
        check_mapped_equivalence(case, root, tools)
    assert changed.is_set()
    assert "Networks are equivalent" in (primitive / "equivalence.log").read_text()
    assert not (root / "composition-proof/proof-record.json").exists()
    assert not (root / "complete-bdd-proof").exists()


@pytest.mark.parametrize("proof_case", [8], indirect=True)
def test_changed_actual_physical_primitive_configuration_is_refused(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Require retained primitive parameters to match the actual cost-bearing physical candidate."""
    case, original, tools, _ = proof_case
    root = tmp_path / "changed-physical-primitive"
    shutil.copytree(original, root)
    path = root / "mapped.json"
    value = json.loads(path.read_text())
    cell = next(
        cell
        for cell in value["modules"]["function_unit"]["cells"].values()
        if cell["type"] == "DSP48E1"
    )
    cell["parameters"]["MREG"] = "1"
    path.write_text(json.dumps(value))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="complete qualified zero-register configuration"):
        read_compositional_mapping_proof(case, root, tools)
    assert _snapshot(root) == before


@pytest.mark.parametrize(
    "name", ["word-proof/word-composition.json", "primitive-proof/primitive-residual.json"]
)
@pytest.mark.parametrize("damage", ["missing", "link", "directory"])
def test_non_regular_component_record_is_refused(
    proof_case: ProofCase, tmp_path: Path, name: str, damage: str
) -> None:
    """Refuse absent, linked or directory-valued actual component records without reacquisition."""
    case, original, tools, _ = proof_case
    root = tmp_path / "non-regular-component"
    shutil.copytree(original, root)
    path = root / "composition-proof" / name
    target = tmp_path / "retained-component-record"
    path.rename(target)
    if damage == "link":
        path.symlink_to(target)
    elif damage == "directory":
        path.mkdir()
    before = _snapshot(root)
    retained = target.read_bytes()
    with pytest.raises(ValueError, match="regular retained artifacts"):
        read_compositional_mapping_proof(case, root, tools)
    assert _snapshot(root) == before and target.read_bytes() == retained


@pytest.mark.parametrize(
    "field", ["wire_bindings_checked", "original_acyclic_order", "product_cells"]
)
def test_word_summary_must_match_recomputed_complete_native_graphs(
    proof_case: ProofCase, tmp_path: Path, field: str
) -> None:
    """Reject rewritten word identities even when all native products and counts remain positive."""
    case, original, tools, _ = proof_case
    root = tmp_path / "changed-word-summary"
    shutil.copytree(original, root)
    path = root / "composition-proof/word-proof/word-composition.json"
    record = json.loads(path.read_bytes())
    if field == "wire_bindings_checked":
        record[field] += 1
    else:
        record[field].pop()
    path.write_text(json.dumps(record))
    before = _snapshot(root)
    with pytest.raises(ValueError, match="actual complete word graphs"):
        read_compositional_mapping_proof(case, root, tools)
    assert _snapshot(root) == before


def test_composition_cannot_reset_the_shared_budget_for_its_bdd_fallback(
    proof_case: ProofCase, tmp_path: Path
) -> None:
    """Exhaust real elapsed time at composition admission and refuse a new full-budget fallback."""
    import subprocess

    case, original, tools, _ = proof_case
    root = tmp_path / "budget-expired-acquisition"
    root.mkdir()
    for path in original.iterdir():
        if path.is_file():
            shutil.copyfile(path, root / path.name)
    script = """from pathlib import Path
import sys
import time
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import check_mapped_equivalence
root = Path(sys.argv[1])
tools = SynthesisToolchain.inspect(root.parent / "inspection", yosys=sys.argv[2])
armed = True
def observe(event, args):
    global armed
    if armed and event == "os.mkdir" and Path(args[0]) == root / "composition-proof":
        armed = False
        time.sleep(10.1)
sys.addaudithook(observe)
try:
    check_mapped_equivalence(SynthesisCase("sin", int(sys.argv[3]), int(sys.argv[3]) // 2),
                             root, tools, timeout=5.0, cohort_seconds=10.0)
except ValueError as error:
    assert "cohort" in str(error)
else:
    raise AssertionError("expired shared budget was reset")
assert not armed
assert not (root / "complete-bdd-proof").exists()
assert not list((root / "composition-proof").glob("*.command.json"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), str(tools.yosys), str(case.data_width)],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
