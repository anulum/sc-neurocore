# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reviewed native Chialvo solver-repair controls

"""Exercise the public repair gate with retained actual native evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import pytest

from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_external_evidence import validate_external
from tools.formal_resource_policy import CHIALVO_JOB, CHIALVO_REPAIR, YICES_RETRY
from tools.formal_solver_repair import validate_solver_repair


@dataclass
class RepairCase:
    """Hold isolated copies of one actual reviewed native repair."""

    repository: Path
    entry: dict[str, object]
    receipt: dict[str, object]
    review: dict[str, object]

    def check(self) -> None:
        """Call the public repair validator with the current copied evidence."""
        validate_solver_repair(
            self.repository,
            Path(CHIALVO_JOB),
            self.entry,
            self.receipt,
            self.review,
        )


@pytest.fixture
def case(tmp_path: Path) -> RepairCase:
    """Copy the maintained production attestation, without generating a proof verdict."""
    source = Path(__file__).resolve().parents[2]
    for name in ("sc_chialvo_map", "sc_chialvo_map_z3memory64"):
        prefix = Path("benchmarks/results/formal") / name
        shutil.copytree(source / prefix, tmp_path / prefix)
    parent = read_object(
        tmp_path / "benchmarks/results/formal/sc_chialvo_map_z3memory64/receipt.json"
    )
    for name in object_fields(parent["source_hashes"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    index = object_fields(read_object(source / "benchmarks/results/formal/index.json")["jobs"])
    entry = object_fields(index[CHIALVO_JOB])
    return RepairCase(
        tmp_path,
        entry,
        read_object(tmp_path / str(entry["receipt"])),
        read_object(tmp_path / str(entry["review"])),
    )


def test_actual_reviewed_native_solver_repair(case: RepairCase) -> None:
    """Accept only the independently reviewed actual error-to-PASS repair."""
    assert case.entry["resource_contract"] == CHIALVO_REPAIR
    case.check()


def test_actual_repair_through_external_gate(case: RepairCase) -> None:
    """Require the full public source, accounting, lineage and non-vacuity gate."""
    solvers = {
        name: Path(executable)
        for name in ("z3", "yices-smt2")
        if (executable := shutil.which(name)) is not None
    }
    yosys = shutil.which("yosys")
    assert yosys is not None
    provenance = (Path(yosys).resolve().parents[1] / "PROVENANCE.txt").read_text()
    result = validate_external(case.repository, Path(CHIALVO_JOB), case.entry, provenance, solvers)
    assert result["native_status"] == "PASS"
    assert object_fields(result["nonvacuity"])["status"] == "NONVACUOUS"


@pytest.mark.parametrize("contract", [YICES_RETRY, None])
def test_repair_requires_its_explicit_contract(case: RepairCase, contract: object) -> None:
    """Ordinary resource retries cannot absorb this original ERROR."""
    case.entry["resource_contract"] = contract
    with pytest.raises(ValueError, match="explicit Chialvo"):
        case.check()


def test_repair_cannot_hide_resource_predecessors(case: RepairCase) -> None:
    """The original crash has its separate bound custody instead of resource history."""
    case.entry["predecessors"] = [{}]
    with pytest.raises(ValueError, match="resource history"):
        case.check()


@pytest.mark.parametrize("change", ["binding", "bytes"])
def test_other_error_receipt_is_refused(case: RepairCase, change: str) -> None:
    """Even refreshed wrapper hashes cannot admit another parent ERROR."""
    entry = object_fields(case.entry["solver_error"])
    if change == "bytes":
        path = case.repository / str(entry["receipt"])
        path.write_bytes(path.read_bytes() + b"\n")
        entry["receipt_sha256"] = digest(path)
    else:
        entry["receipt_sha256"] = "0" * 64
    case.entry["solver_error"] = entry
    with pytest.raises(ValueError, match="exact immutable"):
        case.check()


def test_current_model_must_match_original_error(case: RepairCase) -> None:
    """A model change invalidates the narrowly admitted solver repair."""
    path = case.repository / "hdl/formal/catalogue/sc_chialvo_map.v"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="ERROR sources are stale"):
        case.check()


@pytest.mark.parametrize("field", ["native_status", "classification"])
def test_error_review_must_attest_crash(case: RepairCase, field: str) -> None:
    """A rehashed parent review must still identify the actual solver error."""
    entry = object_fields(case.entry["solver_error"])
    path = case.repository / str(entry["review"])
    review = read_object(path)
    review[field] = "PASS"
    path.write_text(json.dumps(review))
    entry["review_sha256"] = digest(path)
    case.entry["solver_error"] = entry
    with pytest.raises(ValueError, match="attest the solver crash"):
        case.check()


def test_crash_diagnosis_must_be_exact(case: RepairCase) -> None:
    """A different diagnostic cannot support the model-specific repair."""
    entry = object_fields(case.entry["solver_error"])
    path = case.repository / str(entry["diagnostic"])
    path.write_bytes(path.read_bytes() + b"\n")
    entry["diagnostic_sha256"] = digest(path)
    case.entry["solver_error"] = entry
    with pytest.raises(ValueError, match="crash diagnosis"):
        case.check()


def test_variant_must_be_exact(case: RepairCase) -> None:
    """Changing depth invalidates the repair even with an updated outer digest."""
    binding = object_fields(case.entry["configuration"])
    path = case.repository / str(binding["path"])
    original = path.read_text()
    changed = original.replace("depth 4", "depth 5")
    assert changed != original
    path.write_text(changed)
    binding["sha256"] = digest(path)
    case.entry["configuration"] = binding
    with pytest.raises(ValueError, match="reviewed variant"):
        case.check()


def test_portable_solver_identity_is_required(case: RepairCase) -> None:
    """An older or changed Yices binary is outside the exact native repair."""
    tools = object_fields(case.receipt["toolchain"])
    solver = object_fields(tools["solver"])
    solver["sha256"] = "0" * 64
    tools["solver"] = solver
    case.receipt["toolchain"] = tools
    with pytest.raises(ValueError, match="portable Yices"):
        case.check()


@pytest.mark.parametrize("field", ["verdict", "solver_repair"])
def test_new_review_binds_exact_lineage(case: RepairCase, field: str) -> None:
    """The PASS review independently admits this error, diagnosis and variant."""
    case.review[field] = None
    with pytest.raises(ValueError, match="repair lineage"):
        case.check()


@pytest.mark.parametrize("change", ["status", "invocation", "time", "boolean"])
def test_repair_requires_distinct_subsequent_pass(case: RepairCase, change: str) -> None:
    """A failed, reused or earlier invocation cannot become the repaired proof."""
    current = object_fields(case.receipt["measurement"])
    parent_entry = object_fields(case.entry["solver_error"])
    previous = object_fields(
        read_object(case.repository / str(parent_entry["receipt"]))["measurement"]
    )
    if change == "status":
        current["status"] = "ERROR"
    elif change == "invocation":
        current["invocation_id"] = previous["invocation_id"]
    elif change == "time":
        current["start_monotonic_microseconds"] = previous["start_monotonic_microseconds"]
    else:
        current["start_monotonic_microseconds"] = True
    case.receipt["measurement"] = current
    with pytest.raises(ValueError, match="distinct subsequent native PASS"):
        case.check()
