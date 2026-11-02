# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Attested external formal-result gate

"""Validate source-bound, independently attested external formal receipts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tools.formal_evidence_validation import (
    EvidencePolicy,
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
    source_pins,
    validate_measurement,
)
from tools.formal_execution_accounting import validate_execution_accounting
from tools.formal_job_inputs import resolve_job
from tools.formal_log_normalization import validate_public_log
from tools.formal_nonvacuity_evidence import validate_nonvacuity

ML350_POLICY = EvidencePolicy()


def validate_external(
    repository: Path,
    job: Path,
    entry: dict[str, object],
    provenance: str,
    solvers: Mapping[str, Path],
    *,
    policy: EvidencePolicy = ML350_POLICY,
) -> dict[str, object]:
    """Validate a receipt and its independent review against current inputs.

    Parameters
    ----------
    repository : Path
        Current checkout root.
    job : Path
        Tracked SymbiYosys configuration.
    entry : dict of str to object
        Receipt and review paths with their SHA-256 digests.
    provenance : str
        Verified current SDK provenance.
    solvers : mapping of str to Path
        Actual current solver executables.
    policy : EvidencePolicy, optional
        Production ML350 contract; focused native tests use their actual limits.

    Returns
    -------
    dict of str to object
        Explicit attested location, native verdict and actual measurements.

    Raises
    ------
    ValueError
        If any source, tool, task, log, execution or review binding is invalid.
    OSError
        If required files or native tools are unavailable.
    subprocess.SubprocessError
        If the real SBY parser or solver identification fails.
    """
    repository = repository.resolve(strict=True)
    receipt_path = confined_file(repository, entry.get("receipt"))
    receipt_sha = digest(receipt_path)
    require(receipt_sha == entry.get("receipt_sha256"), "Receipt digest differs from the index.")
    receipt = read_object(receipt_path)
    require(
        type(receipt.get("schema_version")) is int and receipt.get("schema_version") == 1,
        "Unknown receipt schema.",
    )
    inputs = resolve_job(repository, job)
    require(
        receipt.get("job") == inputs.job.relative_to(repository).as_posix(),
        "Receipt identifies another formal job.",
    )
    require(receipt.get("source_hashes") == inputs.source_hashes(), "Proof sources are stale.")
    command = object_fields(receipt.get("command"))
    require(command.get("output_option") in ("-d", "--prefix"), "Unknown SBY output option.")
    require(
        command
        == {
            "program": "sby",
            "options": ["-f"],
            "output_option": command["output_option"],
            "directory": inputs.directory.relative_to(repository).as_posix(),
            "argument": inputs.job.relative_to(inputs.directory).as_posix(),
        },
        "Native invocation differs from the unchanged SBY job.",
    )
    task_names = subprocess.run(
        ["sby", "--dumptasks", str(inputs.job)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.splitlines() or ["default"]
    require(
        receipt.get("task_statuses") == dict.fromkeys(task_names, "PASS"),
        "Every declared SBY task must have an actual PASS.",
    )
    tools = object_fields(receipt.get("toolchain"))
    for name, pin in source_pins(provenance).items():
        require(tools.get(name) == pin, f"External {name} differs from the current SDK.")
    for name in ("yosys", "sby"):
        binaries = object_fields(tools.get("binary_sha256"))
        recorded = binaries.get(name)
        require(
            isinstance(recorded, str) and re.fullmatch(r"[0-9a-f]{64}", recorded) is not None,
            f"Used {name} binary digest is missing or invalid.",
        )
    yosys = tools.get("yosys_version")
    require(
        isinstance(yosys, str) and yosys.split(" (", 1)[0] == provenance.split(" (", 1)[0],
        "Used Yosys version differs from the current SDK.",
    )
    sby_versions = [line for line in provenance.splitlines() if line.startswith("SBY v")]
    require(
        len(sby_versions) == 1 and tools.get("sby_version") == sby_versions[0],
        "Used SBY version differs from the current SDK.",
    )
    solver = object_fields(tools.get("solver"))
    solver_name = solver.get("name")
    require(
        isinstance(solver_name, str) and solver_name in solvers,
        "Native solver is not in the current SDK.",
    )
    task_info = object_fields(
        json.loads(
            subprocess.run(
                ["sby", "--dumptaskinfo", str(inputs.job)],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        )
    )
    engine_solver = "yices" if solver_name == "yices-smt2" else solver_name
    engine_groups = [
        group
        for task in task_info.values()
        for group in object_fields(object_fields(task).get("engines")).values()
    ]
    require(
        bool(engine_groups)
        and all(group == [["smtbmc", engine_solver]] for group in engine_groups),
        "Receipt solver differs from the actual SBY task engines.",
    )
    executable = solvers[str(solver_name)]
    require(solver.get("sha256") == digest(executable), "Native solver binary differs.")
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    require(solver.get("version") == version, "Native solver version differs.")
    actual_wall, exception_id, registry_digest = validate_execution_accounting(
        repository, receipt, receipt_sha
    )
    wall, peak = validate_measurement(receipt, policy, actual_wall)
    log_entry = object_fields(receipt.get("log"))
    log = validate_public_log(repository, log_entry)
    markers = re.findall(r"DONE \((PASS|FAIL|ERROR|UNKNOWN|TIMEOUT), rc=(-?\d+)\)", log.read_text())
    require(
        markers == [("PASS", "0")] * len(task_names),
        "Native final DONE markers disagree with the task verdicts.",
    )
    review_path = confined_file(repository, entry.get("review"))
    require(digest(review_path) == entry.get("review_sha256"), "Review digest differs.")
    review = read_object(review_path)
    require(
        type(review.get("schema_version")) is int
        and review.get("schema_version") == 1
        and review.get("verdict") == "CLEAR",
        "Independent external-result review is not CLEAR.",
    )
    require(review.get("receipt_sha256") == receipt_sha, "Review binds another receipt.")
    for field, expected in {
        "raw_log_sha256": log_entry.get("raw_sha256"),
        "normalized_log_sha256": log_entry.get("normalized_sha256"),
        "normalizer_id": log_entry.get("normalizer_id"),
        "normalizer_sha256": log_entry.get("normalizer_sha256"),
        "source_hashes": receipt.get("source_hashes"),
        "execution_exception": exception_id,
        "exception_registry_sha256": registry_digest,
    }.items():
        require(review.get(field) == expected, f"Independent review {field} binding differs.")
    reviewer, producer = review.get("reviewer"), receipt.get("producer")
    require(
        isinstance(reviewer, str)
        and bool(reviewer.strip())
        and isinstance(producer, str)
        and bool(producer.strip())
        and reviewer != producer,
        "Receipt requires a distinct identified independent reviewer.",
    )
    return {
        "location": f"{policy.label}, attested",
        "native_status": "PASS",
        "wall_seconds": wall,
        "peak_memory_bytes": peak,
        "receipt": receipt_path.relative_to(repository).as_posix(),
        "review": review_path.relative_to(repository).as_posix(),
        "nonvacuity": validate_nonvacuity(
            repository,
            inputs,
            entry,
            receipt,
            receipt_sha,
            task_names,
            policy=EvidencePolicy(
                policy.hostname,
                policy.audit_memory_bytes,
                policy.audit_timeout_seconds,
                policy.label,
            ),
        ),
    }


def main(argv: Sequence[str] | None = None, *, policy: EvidencePolicy = ML350_POLICY) -> int:
    """Identify hosted jobs or validate their indexed external evidence.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; process arguments are used when omitted.
    policy : EvidencePolicy, optional
        Explicit native qualification policy. The CLI always uses production
        limits; local contract tests supply their actual machine and limits.

    Returns
    -------
    int
        Zero for a hosted job or qualified external evidence, two for refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--index", type=Path, default=Path("benchmarks/results/formal/index.json"))
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        repository = args.repository.resolve(strict=True)
        index = read_object(confined_file(repository, args.index.as_posix()))
        require(
            type(index.get("schema_version")) is int and index.get("schema_version") == 1,
            "Unknown external-results index schema.",
        )
        jobs = object_fields(index.get("jobs"))
        job = (repository / args.job).resolve(strict=True)
        require(
            job.is_relative_to(repository) and job.suffix == ".sby" and job.is_file(),
            "Formal job must be a .sby file inside the checkout.",
        )
        name = job.relative_to(repository).as_posix()
        if name not in jobs:
            print(json.dumps({"location": "hosted"}))
            return 0
        solvers = {
            name: Path(executable)
            for name in ("z3", "yices-smt2", "cvc5")
            if (executable := shutil.which(name)) is not None
        }
        print(
            json.dumps(
                validate_external(
                    repository,
                    job,
                    object_fields(jobs[name]),
                    args.provenance.read_text(),
                    solvers,
                    policy=policy,
                )
            )
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"External formal evidence refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
