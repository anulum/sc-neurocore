# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real native external-proof receipt contracts

"""Validate real native evidence and refuse isolated mutations of its bindings."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import shutil
import sys

import pytest

from tests.formal_native_evidence_support import (
    JOB,
    NativeEvidence,
    collect_native,
    copy_case,
    copy_identity_conversion,
)
from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_external_evidence import main, validate_external


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> NativeEvidence:
    """Produce one actual bounded proof, including real accounting and closure."""
    return collect_native(tmp_path_factory.mktemp("native-formal-receipt"))


@pytest.fixture
def case(native: NativeEvidence, tmp_path: Path) -> NativeEvidence:
    """Preserve the original native result while isolating each negative control."""
    return copy_case(native, tmp_path)


def _check(case: NativeEvidence, entry: dict[str, object] | None = None) -> dict[str, object]:
    """Exercise the actual public validator against current real proof inputs."""
    return validate_external(
        case.repository,
        JOB,
        case.rebind() if entry is None else entry,
        case.provenance,
        case.solvers,
        policy=case.policy,
    )


def _mutate(case: NativeEvidence, field: str, value: object) -> None:
    """Change exactly one field of the actual receipt for a refusal control."""
    receipt = read_object(case.repository / "receipt.json")
    parts = field.split(".")
    parent = receipt
    for part in parts[:-1]:
        child = object_fields(parent[part])
        parent[part] = child
        parent = child
    parent[parts[-1]] = value
    (case.repository / "receipt.json").write_text(json.dumps(receipt))


def test_actual_native_receipt_and_production_policy_boundary(case: NativeEvidence) -> None:
    """A real local proof qualifies locally and cannot become an ML350 result."""
    result = _check(case)
    assert result["location"] == "local qualification, attested"
    assert result["native_status"] == "PASS"
    assert isinstance(result["wall_seconds"], float) and result["wall_seconds"] > 0
    assert isinstance(result["peak_memory_bytes"], int) and result["peak_memory_bytes"] > 0
    with pytest.raises(ValueError, match="Native .* does not qualify"):
        validate_external(case.repository, JOB, case.rebind(), case.provenance, case.solvers)


def test_missing_host_profile_refuses_actual_native_receipt(case: NativeEvidence) -> None:
    """Reject a missing admitted identity after rebinding an actual native proof."""
    receipt = read_object(case.repository / "receipt.json")
    measurement = object_fields(receipt["measurement"])
    measurement.pop("host_profile")
    receipt["measurement"] = measurement
    (case.repository / "receipt.json").write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="Native host_profile does not qualify"):
        _check(case)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("schema_version", 2, "schema"),
        ("job", "hdl/formal/another.sby", "another formal job"),
        ("source_hashes", {}, "sources are stale"),
        ("command.output_option", "--mode", "output option"),
        ("command.options", ["-f", "--mode", "cover"], "invocation differs"),
        ("task_statuses", {}, "Every declared"),
        ("task_statuses.default", "UNKNOWN", "Every declared"),
        ("toolchain.yosys_source", "0" * 40, "yosys_source"),
        ("toolchain.sby_source", "0" * 40, "sby_source"),
        ("toolchain.binary_sha256.yosys", None, "binary digest"),
        ("toolchain.binary_sha256.sby", "invalid", "binary digest"),
        ("toolchain.yosys_version", None, "Yosys version"),
        ("toolchain.yosys_version", "Yosys 0.1", "Yosys version"),
        ("toolchain.sby_version", "SBY v0.1", "SBY version"),
        ("toolchain.solver.name", None, "solver is not"),
        ("toolchain.solver.name", "unknown", "solver is not"),
        ("toolchain.solver.sha256", "0" * 64, "solver binary"),
        ("toolchain.solver.version", "wrong version", "solver version"),
        ("measurement.host_profile", "unadmitted-profile", "host_profile"),
        ("measurement.status", "TIMEOUT", "status"),
        ("measurement.unit_result", "oom-kill", "systemd exit facts"),
        ("measurement.unit_accounting", None, "accounting is missing"),
        ("measurement.unit_exit_code", False, "systemd exit facts"),
        ("measurement.native_exit_source", "sby-final-DONE", "systemd exit facts"),
        ("measurement.wall_clock_source", "unknown", "clock is missing"),
        ("measurement.start_monotonic_microseconds", 0, "clock stamps"),
        ("measurement.stop_monotonic_microseconds", False, "clock stamps"),
        ("measurement.active_state", "active", "active_state"),
        ("measurement.process_tree_closed", False, "process tree"),
        ("measurement.process_tree_closed", 1, "process tree"),
        ("measurement.cgroup_processes", [123], "cgroup_processes"),
        ("measurement.native_exit_code", 1, "exit code"),
        ("measurement.native_exit_code", False, "exit code"),
        ("measurement.main_pid", 123, "main_pid"),
        ("measurement.memory_limit_bytes", 1, "memory_limit_bytes"),
        ("measurement.swap_limit_bytes", 1, "swap_limit_bytes"),
        ("measurement.time_limit_seconds", 7200, "time_limit_seconds"),
        ("measurement.invocation_id", None, "invocation identity"),
        ("measurement.invocation_id", "invalid", "invocation identity"),
        ("measurement.oom_events.oom", 1, "oom"),
        ("measurement.oom_events.oom_kill", 1, "oom_kill"),
        ("measurement.oom_events.oom_group_kill", 1, "oom_group_kill"),
        ("measurement.wall_seconds", "1", "numeric"),
        ("measurement.wall_seconds", True, "numeric"),
        ("measurement.wall_seconds", float("nan"), "finite"),
        ("measurement.wall_seconds", float("inf"), "finite"),
        ("measurement.wall_seconds", 0, "monotonic stamps"),
        ("measurement.wall_seconds", 61, "monotonic stamps"),
        ("measurement.peak_memory_bytes", 0, "peak is outside"),
        ("measurement.peak_memory_bytes", 128 * 1024**2 + 1, "peak is outside"),
        ("measurement.peak_memory_bytes", 1.5, "peak is outside"),
        ("log.normalized_sha256", "0" * 64, "log digest"),
        ("log.raw_sha256", None, "custody digest"),
        ("log.raw_sha256", "invalid", "custody digest"),
        ("log.normalizer_id", "unknown", "normalizer"),
        ("log.normalizer_sha256", "0" * 64, "normalizer source"),
    ],
)
def test_real_receipt_mutations_refuse(
    case: NativeEvidence, field: str, value: object, reason: str
) -> None:
    """Refuse each changed native fact after refreshing outer test-only digests."""
    _mutate(case, field, value)
    with pytest.raises(ValueError, match=reason):
        _check(case)


def test_changed_real_source_and_log_markers_refuse(case: NativeEvidence) -> None:
    """Actual changed source bytes and contradictory final log markers invalidate evidence."""
    source = case.repository / JOB.with_suffix(".v")
    source.write_text(source.read_text() + "\n// Changed proof input.\n")
    with pytest.raises(ValueError, match="sources are stale"):
        _check(case)


@pytest.mark.parametrize("marker", ["DONE (FAIL, rc=2)", "DONE (PASS, rc=1)", "missing"])
def test_real_log_marker_changes_refuse(case: NativeEvidence, marker: str) -> None:
    """A rehashed log cannot contradict its claimed successful task verdict."""
    log = case.repository / "public-native.log"
    log.write_text(log.read_text().replace("DONE (PASS, rc=0)", marker))
    _mutate(case, "log.normalized_sha256", digest(log))
    with pytest.raises(ValueError, match="DONE markers"):
        _check(case)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "digest",
        "verdict",
        "schema",
        "receipt",
        "actor",
        "empty",
        "producer",
        "producer_empty",
        "raw_binding",
        "normalized_binding",
        "normalizer_binding",
        "normalizer_source_binding",
        "source_binding",
    ],
)
def test_missing_or_changed_independent_review_refuses(case: NativeEvidence, change: str) -> None:
    """A native PASS cannot substitute for an independent receipt-bound review."""
    entry = case.rebind()
    if change == "missing":
        entry.pop("review")
    elif change == "digest":
        entry["review_sha256"] = "0" * 64
    else:
        review = read_object(case.repository / "review.json")
        field, value = {
            "verdict": ("verdict", "OPEN"),
            "schema": ("schema_version", 2),
            "receipt": ("receipt_sha256", "0" * 64),
            "actor": ("reviewer", "actual native receipt test producer"),
            "empty": ("reviewer", ""),
            "producer": ("reviewer", None),
            "producer_empty": ("reviewer", "independent test actor"),
            "raw_binding": ("raw_log_sha256", "0" * 64),
            "normalized_binding": ("normalized_log_sha256", "0" * 64),
            "normalizer_binding": ("normalizer_id", "unknown"),
            "normalizer_source_binding": ("normalizer_sha256", "0" * 64),
            "source_binding": ("source_hashes", {}),
        }[change]
        review[field] = value
        if change == "producer_empty":
            _mutate(case, "producer", "")
            entry["receipt_sha256"] = digest(case.repository / "receipt.json")
            review["receipt_sha256"] = entry["receipt_sha256"]
        (case.repository / "review.json").write_text(json.dumps(review))
        entry["review_sha256"] = digest(case.repository / "review.json")
    with pytest.raises(ValueError):
        _check(case, entry)


@pytest.mark.parametrize("path", [None, "/etc/hostname", "../outside", "."])
def test_evidence_path_confinement(case: NativeEvidence, path: object) -> None:
    """Refuse absent, absolute, escaped and directory evidence references."""
    (case.repository.parent / "outside").write_text("outside\n")
    entry = case.rebind()
    entry["review"] = path
    with pytest.raises(ValueError):
        _check(case, entry)


def test_index_receipt_digest_and_object_contracts(case: NativeEvidence) -> None:
    """Refuse altered receipt bytes and malformed object representations."""
    entry = case.rebind()
    entry["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Receipt digest"):
        _check(case, entry)
    invalid_objects: tuple[object, ...] = ([], {1: "not a JSON object key"})
    for value in invalid_objects:
        with pytest.raises(ValueError, match="object with string keys"):
            object_fields(value)


@pytest.mark.parametrize("provenance", ["missing", "duplicate", "missing_version"])
def test_sdk_provenance_refusals(case: NativeEvidence, provenance: str) -> None:
    """Missing or ambiguous current SDK identities cannot attest an old result."""
    if provenance == "missing":
        case.provenance = "unidentified SDK\n"
    elif provenance == "duplicate":
        case.provenance += case.provenance
    else:
        case.provenance = "\n".join(
            line for line in case.provenance.splitlines() if not line.startswith("SBY v")
        )
    with pytest.raises(ValueError, match="SDK|SBY version"):
        _check(case)


def test_cli_hosted_classification_and_refused_local_result(case: NativeEvidence) -> None:
    """Exercise the real module entry and production refusal without replacing native tools."""
    index = case.repository / "index.json"
    provenance = case.repository / "PROVENANCE.txt"
    provenance.write_text(case.provenance)
    index.write_text(json.dumps({"schema_version": 1, "jobs": {}}))
    args = [
        str(JOB),
        "--repository",
        str(case.repository),
        "--index",
        "index.json",
        "--provenance",
        str(provenance),
    ]
    assert main(args) == 0
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "-m",
        "tools.formal_external_evidence",
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {"location": "hosted"}
    index.write_text(json.dumps({"schema_version": 1, "jobs": {JOB.as_posix(): case.rebind()}}))
    assert main(args, policy=case.policy) == 0
    assert main(args) == 2
    index.write_text(json.dumps({"schema_version": 2, "jobs": {}}))
    assert main(args) == 2


def test_actual_other_solver_cannot_attest_z3_job(case: NativeEvidence) -> None:
    """A real available Yices identity cannot attest a job whose native engine is Z3."""
    name = "yices-smt2"
    executable = shutil.which(name)
    assert executable is not None, "The shared formal SDK must provide Yices."
    solver = Path(executable)
    case.solvers[name] = solver
    _mutate(
        case,
        "toolchain.solver",
        {
            "name": name,
            "sha256": digest(solver),
            "version": subprocess.check_output([str(solver), "--version"], text=True).strip(),
        },
    )
    with pytest.raises(ValueError, match="actual SBY task engines"):
        _check(case)


@pytest.fixture
def retry(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Copy the actual reviewed ML350 resource-to-PASS sequence for gate controls."""
    repository = Path(__file__).resolve().parents[2]
    for name in ("sc_chaotic_map", "sc_chaotic_map_z3primary40", "sc_chaotic_map_z3memory64"):
        prefix = Path("benchmarks/results/formal") / name
        shutil.copytree(repository / prefix, tmp_path / prefix)
    parent = read_object(
        tmp_path / "benchmarks/results/formal/sc_chaotic_map_z3primary40/receipt.json"
    )
    for name in object_fields(parent["source_hashes"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository / name).read_bytes())
    index = object_fields(read_object(repository / "benchmarks/results/formal/index.json")["jobs"])
    entry = object_fields(index["hdl/formal/catalogue/sc_chaotic_map.sby"])
    copy_identity_conversion(repository, tmp_path, entry)
    return tmp_path, entry


def _check_retry(retry: tuple[Path, dict[str, object]]) -> dict[str, object]:
    """Validate the retained production sequence using actual current SDK tools."""
    repository, entry = retry
    yosys = shutil.which("yosys")
    assert yosys is not None
    provenance = (Path(yosys).resolve().parents[1] / "PROVENANCE.txt").read_text()
    solvers = {
        name: Path(executable)
        for name in ("z3", "yices-smt2")
        if (executable := shutil.which(name)) is not None
    }
    return validate_external(
        repository, Path("hdl/formal/catalogue/sc_chaotic_map.sby"), entry, provenance, solvers
    )


def test_actual_resource_retry_pass_and_nonvacuity(
    retry: tuple[Path, dict[str, object]],
) -> None:
    """The public gate accepts actual MEMORY40, TIMEOUT64, Yices PASS and audit."""
    result = _check_retry(retry)
    assert result["native_status"] == "PASS"
    assert object_fields(result["nonvacuity"])["status"] == "NONVACUOUS"


@pytest.mark.parametrize("change", ["chain", "review", "invocation", "clock"])
def test_resource_retry_pass_lineage_refusals(
    retry: tuple[Path, dict[str, object]], change: str
) -> None:
    """A rehashed negative fixture cannot hide missing or overlapping retry lineage."""
    repository, entry = retry
    history = entry["predecessors"]
    assert isinstance(history, list)
    review_path = repository / str(entry["review"])
    review = read_object(review_path)
    if change == "chain":
        entry["predecessors"] = history[:-1]
        reason = "admitted predecessor chain"
    elif change == "review":
        review["predecessors"] = []
        reason = "bind the predecessor history"
    else:
        path = repository / str(entry["receipt"])
        receipt = read_object(path)
        current = object_fields(receipt["measurement"])
        last = object_fields(history[-1])
        previous = object_fields(read_object(repository / str(last["receipt"]))["measurement"])
        if change == "invocation":
            current["invocation_id"] = previous["invocation_id"]
        else:
            duration = int(str(current["stop_monotonic_microseconds"])) - int(
                str(current["start_monotonic_microseconds"])
            )
            current["start_monotonic_microseconds"] = previous["start_monotonic_microseconds"]
            current["stop_monotonic_microseconds"] = (
                int(str(current["start_monotonic_microseconds"])) + duration
            )
        receipt["measurement"] = current
        path.write_text(json.dumps(receipt))
        entry["receipt_sha256"] = digest(path)
        review["receipt_sha256"] = digest(path)
        # Converted native facts are now bound before the later lineage check.
        reason = "Public identity conversion artifact binding differs"
    review["reviewer"] = "Negative lineage control; not a production attestation"
    review_path.write_text(json.dumps(review))
    entry["review_sha256"] = digest(review_path)
    with pytest.raises(ValueError, match=reason):
        _check_retry(retry)
