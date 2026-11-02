# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independently attested assertion-enable evidence

"""Validate a separate enable audit without rewriting its primary receipt."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
import subprocess

from tools.formal_evidence_validation import (
    REFERENCE_HOST_PROFILE,
    EvidencePolicy,
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
    validate_measurement,
)
from tools.formal_execution_accounting import validate_execution_accounting
from tools.formal_job_inputs import FormalJobInputs
from tools.formal_log_normalization import validate_public_log
from tools.formal_nonvacuity import native_configuration
from tools.formal_receipt_identity import executed_configuration

ML350_AUDIT_POLICY = EvidencePolicy(
    REFERENCE_HOST_PROFILE, memory_bytes=8 * 1024**3, timeout_seconds=1800
)


def _task(
    repository: Path,
    name: str | None,
    value: object,
    inputs: FormalJobInputs,
    configuration_path: Path,
    primary_pass: bool,
) -> str:
    """Check native task options, model custody and all enable witnesses."""
    task = object_fields(value)
    native = subprocess.run(
        ["sby", "--dumpcfg", str(configuration_path), *([] if name is None else [name])],
        cwd=inputs.directory,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    configuration = native_configuration(native)
    require(task.get("configuration") == configuration, "Native audit configuration differs.")
    options = object_fields(configuration.get("options"))
    report = object_fields(task.get("report"))
    depth_text = options.get("depth")
    require(isinstance(depth_text, str) and depth_text.isdecimal(), "Audit depth is missing.")
    depth = int(str(depth_text))
    require(
        depth > 0
        and type(report.get("depth")) is int
        and report.get("depth") == depth
        and report.get("mode") == options.get("mode"),
        "Audit mode or depth differs from the current native task.",
    )
    require(
        type(report.get("schema_version")) is int
        and report.get("schema_version") == 1
        and report.get("check") == "assertion-enable non-vacuity"
        and report.get("verdict") == "NONVACUOUS",
        "Assertion-enable audit is not NONVACUOUS.",
    )
    custody = object_fields(task.get("model_custody"))
    model_sha = report.get("design_prep_sha256")
    for field in ("design_prep_sha256", "config_sha256", "prepared_json_sha256", "sby_sha256"):
        value = report.get(field)
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Audit {field} is missing or invalid.",
        )
    require(custody.get("audited_sha256") == model_sha, "Audited model digest differs.")
    origin = custody.get("origin")
    require(
        (origin == "retained original" and custody.get("primary_sha256") == model_sha)
        or (
            origin == "regenerated, original not retained" and custody.get("primary_sha256") is None
        ),
        "Primary prepared-model custody does not qualify.",
    )
    require(
        report.get("implementation_sha256")
        == digest(Path(__file__).with_name("formal_nonvacuity.py")),
        "Audit implementation is stale.",
    )
    if options.get("mode") == "cover":
        require(primary_pass, "Original cover requires an actual primary PASS.")
        covers = report.get("original_cover_goals")
        require(
            isinstance(covers, list)
            and bool(covers)
            and all(isinstance(goal, str) and bool(goal) for goal in covers)
            and len(set(covers)) == len(covers),
            "Audit cover goals must be nonempty and unique.",
        )
    else:
        goals = object_fields(report.get("assertions"))
        require(
            bool(goals)
            and all(
                bool(goal) and isinstance(identity, str) and bool(identity)
                for goal, identity in goals.items()
            )
            and len(set(goals.values())) == len(goals),
            "Audit assertion identities must be nonempty and unique.",
        )
        require(options.get("mode") in ("bmc", "prove"), "Unsupported audit mode.")
        reached = object_fields(report.get("first_reached_steps"))
        require(
            set(reached) == set(goals)
            and all(type(step) is int and 0 <= step < depth for step in reached.values())
            and report.get("assertion_enable_status") == "REACHED"
            and report.get("unreached_assertions") == [],
            "An assertion enable is UNREACHED within the native depth.",
        )
        require(
            report.get("transform_sha256")
            == digest(Path(__file__).with_name("formal_assertion_enables.py")),
            "Audit transform is stale.",
        )
        require(
            report.get("native_status") == "PASS"
            and type(report.get("native_exit_code")) is int
            and report.get("native_exit_code") == 0,
            "Native enable cover did not PASS.",
        )
        log_entry = object_fields(task.get("log"))
        log = validate_public_log(repository, log_entry).read_text()
        require(
            report.get("auxiliary_log_sha256") == log_entry.get("raw_sha256"),
            "Audit log binding differs.",
        )
        require(
            re.findall(r"DONE \((PASS|FAIL|ERROR|UNKNOWN|TIMEOUT), rc=(-?\d+)\)", log)
            == [("PASS", "0")],
            "Native audit DONE marker differs.",
        )
        for goal, step in reached.items():
            require(
                re.search(
                    r"Reached cover statement in step "
                    + str(step)
                    + r" at [^\n]*\b"
                    + re.escape(goal)
                    + r"\b",
                    log,
                )
                is not None,
                "Native audit log lacks an enable witness.",
            )
    return str(model_sha)


def validate_nonvacuity(
    repository: Path,
    inputs: FormalJobInputs,
    entry: dict[str, object],
    receipt: dict[str, object],
    receipt_sha256: str,
    task_names: Sequence[str],
    *,
    policy: EvidencePolicy = ML350_AUDIT_POLICY,
) -> dict[str, object]:
    """Require source-bound, measured and independently reviewed enable evidence.

    Parameters
    ----------
    repository : Path
        Checkout containing the indexed evidence and current formal sources.
    inputs : FormalJobInputs
        Native source closure already resolved for the primary receipt.
    entry : dict of str to object
        Index entry with separate nonvacuity and nonvacuity_review paths and hashes.
    receipt : dict of str to object
        Validated immutable primary receipt.
    receipt_sha256 : str
        Digest of the unchanged primary receipt bytes.
    task_names : sequence of str
        Every declared native primary task, with default for an unnamed task.
    policy : EvidencePolicy, optional
        Separate audit resource contract; production uses ML350 8 GiB/1800 s.

    Returns
    -------
    dict of str to object
        Audit location, qualified status and its actual separate measurements.

    Raises
    ------
    ValueError
        If model, source, implementation, witness, resource or review bindings fail.
    OSError
        If required actual evidence files or native tools are unavailable.
    subprocess.SubprocessError
        If the current native task configuration cannot be resolved.
    """
    artifact_path = confined_file(repository, entry.get("nonvacuity"))
    artifact_sha = digest(artifact_path)
    require(artifact_sha == entry.get("nonvacuity_sha256"), "Nonvacuity artifact digest differs.")
    artifact = read_object(artifact_path)
    require(
        type(artifact.get("schema_version")) is int and artifact.get("schema_version") == 1,
        "Unknown nonvacuity schema.",
    )
    require(
        artifact.get("receipt_sha256") == receipt_sha256, "Audit binds another primary receipt."
    )
    configuration_path = executed_configuration(repository, inputs, entry, receipt)
    sources = object_fields(receipt.get("source_hashes"))
    require(artifact.get("source_hashes") == sources, "Audit sources are stale.")
    require(
        artifact.get("toolchain") == receipt.get("toolchain"),
        "Audit tools differ from the primary.",
    )
    measurement = object_fields(artifact.get("measurement"))
    primary_measurement = object_fields(receipt.get("measurement"))
    require(
        isinstance(measurement.get("unit"), str)
        and bool(measurement["unit"])
        and measurement["unit"] != primary_measurement.get("unit")
        and measurement.get("invocation_id") != primary_measurement.get("invocation_id"),
        "Audit must have a separate native service invocation.",
    )
    require(
        measurement.get("unit_accounting") == "retained",
        "Audit execution accounting was not retained.",
    )
    wall, _exception, _registry = validate_execution_accounting(repository, artifact, artifact_sha)
    wall, peak = validate_measurement(artifact, policy, wall)
    tasks = object_fields(artifact.get("tasks"))
    require(bool(tasks) and set(tasks) == set(task_names), "Audit must cover every primary task.")
    binaries = object_fields(object_fields(artifact.get("toolchain")).get("binary_sha256"))
    require(
        all(
            object_fields(object_fields(task).get("report")).get("sby_sha256")
            == binaries.get("sby")
            for task in tasks.values()
        ),
        "Audit SBY binary differs from the measured toolchain.",
    )
    declared = subprocess.run(
        ["sby", "--dumptasks", str(configuration_path)],
        cwd=inputs.directory,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.splitlines()
    models = {
        name: _task(
            repository,
            name if name in declared else None,
            task,
            inputs,
            configuration_path,
            primary_measurement.get("status") == "PASS",
        )
        for name, task in tasks.items()
    }
    review_path = confined_file(repository, entry.get("nonvacuity_review"))
    require(
        digest(review_path) == entry.get("nonvacuity_review_sha256"),
        "Nonvacuity review digest differs.",
    )
    review = read_object(review_path)
    require(
        type(review.get("schema_version")) is int
        and review.get("schema_version") == 1
        and review.get("verdict") == "CLEAR",
        "Independent nonvacuity review is not CLEAR.",
    )
    for field, expected in {
        "artifact_sha256": artifact_sha,
        "receipt_sha256": receipt_sha256,
        "source_hashes": sources,
        "prepared_model_sha256": models,
    }.items():
        require(review.get(field) == expected, f"Nonvacuity review {field} binding differs.")
    producer, reviewer = artifact.get("producer"), review.get("reviewer")
    require(
        isinstance(producer, str)
        and bool(producer.strip())
        and isinstance(reviewer, str)
        and bool(reviewer.strip())
        and producer != reviewer,
        "Nonvacuity evidence requires a distinct identified independent reviewer.",
    )
    return {
        "status": "NONVACUOUS",
        "artifact": artifact_path.relative_to(repository).as_posix(),
        "wall_seconds": wall,
        "peak_memory_bytes": peak,
    }
