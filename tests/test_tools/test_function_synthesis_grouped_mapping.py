# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual production arithmetic mapping admission

"""Exercise actual joint or individual word proofs and readonly refusal of damaged evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_proof import read_compositional_mapping_proof
from tools.function_synthesis_records import SynthesisCase
from tools.function_synthesis_toolchain import SynthesisToolchain
from tools.function_synthesis_yosys import (
    FunctionSynthesisMeasurement,
    read_function_synthesis_measurement,
    run_function_synthesis,
)


@pytest.fixture(scope="module")
def grouped_measurement(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement]:
    """Acquire a fresh complete production measurement, including its selected native proof path."""
    root = tmp_path_factory.mktemp("actual-complete-grouped-measurement")
    tools = SynthesisToolchain.inspect(root / "inspection")
    case = SynthesisCase("hypot", 16, 8)
    measurement = run_function_synthesis(
        case, root / "measurement", tools, synthesis_timeout=900, proof_timeout=90
    )
    return case, root / "measurement", tools, measurement


def test_grouped_production_measurement_preserves_joint_proofs_and_original_native_artifacts(
    grouped_measurement: tuple[
        SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement
    ],
) -> None:
    """Read the actual word-proof strategy without inventing individual or joint receipts."""
    case, root, tools, measurement = grouped_measurement
    before = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    proof = read_compositional_mapping_proof(case, root, tools)
    assert read_function_synthesis_measurement(case, root, tools) == measurement
    record = json.loads((root / "composition-proof/proof-record.json").read_text())
    assert record["schema_version"] == "sc-neurocore.function-composition.v3"
    products = list((root / "composition-proof/word-proof").glob("product-*.json"))
    if record["word"].get("schema_version") == "sc-neurocore.word-composition.v2":
        assert record["word"]["products_proven"] == 0
        assert record["word"]["products_covered_jointly"] == 2
        assert not products and len(proof.word_work) == 1
    else:
        assert record["word"]["products_proven"] == 2
        assert len(products) == 2 and not proof.word_work
        assert not (root / "composition-proof/word-proof/group-work.json").exists()
        assert not (root / "composition-proof/word-readback-work.json").exists()
    assert measurement.resources.dsp_blocks > 0
    assert before == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in before
    }


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("producer-copy", "producer work differs from its original record"),
        ("combined-time", "arithmetic work exhausted the shared case budget"),
    ],
)
def test_grouped_reader_preserves_original_work_and_shared_producer_reader_budget(
    grouped_measurement: tuple[
        SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement
    ],
    tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    """Reject changed producer evidence and exhaustion of the actual native proof's shared budget."""
    case, original, tools, _ = grouped_measurement
    original_before = _snapshot(original)
    root = tmp_path / "changed-work-ledger"
    shutil.copytree(original, root, copy_function=os.link)
    composition = root / "composition-proof"
    word = object_mapping(
        json.loads((composition / "word-proof/word-composition.json").read_text())
    )
    if word.get("schema_version") != "sc-neurocore.word-composition.v2":
        if damage == "producer-copy":
            path = composition / "word-proof/product-000.json"
            product = object_mapping(json.loads(path.read_text()))
            product["original_cell"] = "changed_original_multiplier"
            path.unlink()
            path.write_text(json.dumps(product, sort_keys=True, indent=2) + "\n")
            bounded_tools = tools
            message = "complete retained original multiplication was not proved"
        else:
            bounded_tools = replace(tools, cohort_deadline=time.monotonic() - 1)
            message = "exhausted the shared case budget"
        changed_before = _snapshot(root)
        with pytest.raises(ValueError, match=message):
            read_compositional_mapping_proof(case, root, bounded_tools)
        assert _snapshot(root) == changed_before
        assert _snapshot(original) == original_before
        return
    producer_path = composition / "word-proof/group-work.json"
    readback_path = composition / "word-readback-work.json"
    producer = object_mapping(json.loads(producer_path.read_text()))
    readback = object_mapping(json.loads(readback_path.read_text()))
    if damage == "producer-copy":
        producer["elapsed_seconds"] = 0.0
        assert readback["producer"] != producer
    else:
        producer["elapsed_seconds"] = 1700.0
        readback["producer"] = producer
        reader = object_mapping(readback["reader"])
        reader["elapsed_seconds"] = 1700.0
        readback["reader"] = reader
    for path, payload in ((producer_path, producer), (readback_path, readback)):
        path.unlink()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    changed_before = _snapshot(root)
    with pytest.raises(ValueError, match=message):
        read_compositional_mapping_proof(case, root, tools)
    assert _snapshot(root) == changed_before
    assert _snapshot(original) == original_before


def _snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """Read retained acquisition bytes and timestamps without altering their custody."""
    return {
        path.relative_to(root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "damage",
    [
        "group-schema",
        "member",
        "root-width",
        "root-high-bit",
        "ring",
        "frontier",
        "joint-count",
        "producer-work",
        "reader-work",
        "work-deadline",
        "schema",
        "tool",
        "source",
    ],
)
def test_grouped_production_reader_refuses_changed_complete_certificates_without_reacquisition(
    grouped_measurement: tuple[
        SynthesisCase, Path, SynthesisToolchain, FunctionSynthesisMeasurement
    ],
    tmp_path: Path,
    damage: str,
) -> None:
    """Validate native word schemas, complete products or groups, bindings and bounded work."""
    case, original, tools, _ = grouped_measurement
    before = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in original.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "damaged"
    shutil.copytree(original, root, copy_function=os.link)
    composition = root / "composition-proof"
    word = object_mapping(
        json.loads((composition / "word-proof/word-composition.json").read_text())
    )
    grouped = word.get("schema_version") == "sc-neurocore.word-composition.v2"
    if not grouped and damage == "work-deadline":
        unchanged = _snapshot(root)
        with pytest.raises(ValueError, match="exhausted the shared case budget"):
            read_compositional_mapping_proof(
                case, root, replace(tools, cohort_deadline=time.monotonic() - 1)
            )
        assert _snapshot(root) == unchanged
        assert _snapshot(original) == {
            p.relative_to(original).as_posix(): value for p, value in before.items()
        }
        return
    if damage == "source":
        path = root / "unit.v"
        data = path.read_bytes() + b"\n"
    elif not grouped:
        if damage in {"schema", "tool", "reader-work"}:
            path = composition / "proof-record.json"
        elif damage in {"member", "root-width", "root-high-bit", "ring"}:
            path = composition / "word-proof/product-000.json"
        elif damage == "frontier":
            path = composition / "word-proof/wire-bindings.json"
        else:
            path = composition / "word-proof/word-composition.json"
        record = json.loads(path.read_text())
        if damage == "schema":
            record["schema_version"] = "sc-neurocore.function-composition.v1"
        elif damage == "tool":
            record["toolchain_sha256"] = "0" * 64
        elif damage == "group-schema":
            record["schema_version"] = "sc-neurocore.word-composition.v2"
        elif damage == "member":
            record["original_cell"] = "changed_original_multiplier"
        elif damage == "root-width":
            record["complete_product_bits"] -= 1
        elif damage == "root-high-bit":
            record["mapped"][-1]["coefficient"] ^= 1 << (record["complete_product_bits"] - 1)
        elif damage == "ring":
            record["difference"] = [{"coefficient": 1, "factors": []}]
        elif damage == "frontier":
            other = next(i for i, pair in enumerate(record) if pair[1] != record[0][1])
            record[0][1], record[other][1] = record[other][1], record[0][1]
        elif damage == "joint-count":
            record["products_proven"] -= 1
        elif damage == "producer-work":
            record["product_cells"].pop()
        else:
            record["word"]["wire_bindings_checked"] += 1
        data = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    else:
        if damage in {"schema", "tool"}:
            path = composition / "proof-record.json"
        elif damage == "reader-work":
            path = composition / "word-readback-work.json"
        elif damage in {"producer-work", "work-deadline"}:
            path = composition / "word-proof/group-work.json"
        else:
            path = composition / "word-proof/word-composition.json"
        record = json.loads(path.read_text())
        if damage == "schema":
            record["schema_version"] = "sc-neurocore.function-composition.v1"
        elif damage == "tool":
            record["toolchain_sha256"] = "0" * 64
        elif damage in {"producer-work", "reader-work"}:
            (record["reader"] if damage == "reader-work" else record)["operation_calls"]["add"] += 1
        elif damage == "work-deadline":
            record["elapsed_seconds"] = 3301
        elif damage == "joint-count":
            record["products_covered_jointly"] += 1
        else:
            group = record["arithmetic_groups"][0]
            if damage == "group-schema":
                group["schema_version"] = "unqualified-group"
            elif damage == "member":
                group["source_members"].pop()
            elif damage == "root-width":
                group["roots"][0]["width"] -= 1
            elif damage == "root-high-bit":
                bits = group["roots"][0]["mapped_bits"]
                bits[-1] = "1" if bits[-1] == "0" else "0"
            elif damage == "ring":
                group["roots"][0]["ring"] = "Z/(2^1Z)"
            else:
                group["frontier"][0][1], group["frontier"][1][1] = (
                    group["frontier"][1][1],
                    group["frontier"][0][1],
                )
        data = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    path.unlink()
    path.write_bytes(data)
    damaged = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError):
        read_compositional_mapping_proof(case, root, tools)
    assert damaged == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in damaged
    }
    assert before == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in before
    }
