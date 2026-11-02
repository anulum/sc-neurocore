# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared source, solver and independent review bindings

"""Bind native formal attempts to their actual source, solver and review."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import subprocess

from tools.formal_evidence_validation import (
    confined_file,
    digest,
    object_fields,
    read_object,
    require,
    source_pins,
)
from tools.formal_job_inputs import FormalJobInputs, resolve_job


def solver_configuration(text: str, solver: str) -> str:
    """Change only SMTBMC solver tokens inside the original engines section.

    Parameters
    ----------
    text : str
        Exact current tracked SBY configuration, including all comments.
    solver : str
        Explicit current SDK executable, either z3 or yices-smt2.

    Returns
    -------
    str
        Configuration with modes, depths, scripts, files and other bytes kept.

    Raises
    ------
    ValueError
        If a solver or engine line is outside the approved SMTBMC variant.
    """
    require(solver in ("z3", "yices-smt2"), "Unsupported solver-only configuration.")
    token = "yices" if solver == "yices-smt2" else "z3"
    pattern = re.compile(
        r"([ \t]*(?:[^ \t:]+:[ \t]*)?smtbmc)(?:([ \t]+)(z3|yices))?"
        r"([ \t]*(?:#[^\r\n]*)?)(\r?\n)?"
    )
    engines = False
    changed = 0
    result = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            engines = stripped == "[engines]"
        elif engines and stripped and not stripped.startswith("#"):
            match = pattern.fullmatch(line)
            if match is None:
                raise ValueError("Unsupported solver-only engine line.")
            line = match[1] + (match[2] or " ") + token + match[4] + (match[5] or "")
            changed += 1
        result.append(line)
    require(changed > 0, "Solver-only configuration has no SMTBMC engines.")
    return "".join(result)


def executed_configuration(
    repository: Path,
    inputs: FormalJobInputs,
    entry: Mapping[str, object],
    receipt: Mapping[str, object],
) -> Path:
    """Bind either unchanged sources or an exact retained solver-only variant.

    Parameters
    ----------
    repository : Path
        Resolved checkout containing the current model and retained variant.
    inputs : FormalJobInputs
        Current canonical source closure reported by the real SBY parser.
    entry : mapping of str to object
        Digest-bound index entry with optional executed configuration custody.
    receipt : mapping of str to object
        Actual measured source digests and solver identity.

    Returns
    -------
    Path
        Actual configuration whose tasks and engines must be parsed.

    Raises
    ------
    ValueError
        If any HDL source differs or a variant changes more than its solver.
    """
    expected = inputs.source_hashes()
    configuration = inputs.job
    if "configuration" in entry:
        binding = object_fields(entry["configuration"])
        configuration = confined_file(repository, binding.get("path"))
        require(
            configuration.suffix == ".sby" and digest(configuration) == binding.get("sha256"),
            "Retained solver configuration digest differs.",
        )
        solver = object_fields(object_fields(receipt.get("toolchain")).get("solver")).get("name")
        require(isinstance(solver, str), "Solver-only configuration identity is missing.")
        require(
            configuration.read_bytes()
            == solver_configuration(inputs.job.read_bytes().decode("utf-8"), str(solver)).encode(
                "utf-8"
            ),
            "Executed configuration changes more than the admitted solver.",
        )
        expected[inputs.job.relative_to(repository).as_posix()] = digest(configuration)
    require(receipt.get("source_hashes") == expected, "Proof sources are stale.")
    return configuration


def validate_identity(
    repository: Path,
    job: Path,
    entry: Mapping[str, object],
    receipt: dict[str, object],
    provenance: str,
    solvers: Mapping[str, Path],
) -> tuple[FormalJobInputs, list[str], str]:
    """Check unchanged model custody, invocation and actual pinned tool identity.

    Parameters
    ----------
    repository : Path
        Resolved current checkout root.
    job : Path
        Current canonical SBY configuration.
    entry : mapping of str to object
        Trusted index entry and any retained solver-only configuration.
    receipt : dict of str to object
        Digest-bound actual native measurement receipt.
    provenance : str
        Current verified SDK provenance.
    solvers : mapping of str to Path
        Current actual solver executables.

    Returns
    -------
    tuple of FormalJobInputs, list of str and str
        Canonical source closure, declared tasks and verified solver name.

    Raises
    ------
    ValueError
        If any source, command, task engine or SDK identity differs.
    OSError
        If the parser, native solver or retained files are unavailable.
    subprocess.SubprocessError
        If an actual tool query fails or exceeds its finite timeout.
    """
    inputs = resolve_job(repository, job)
    require(
        receipt.get("job") == inputs.job.relative_to(repository).as_posix(),
        "Receipt identifies another formal job.",
    )
    configuration = executed_configuration(repository, inputs, entry, receipt)
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
                ["sby", "--dumptaskinfo", str(configuration)],
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
    return inputs, task_names, str(solver_name)


def validate_review(
    repository: Path,
    entry: Mapping[str, object],
    receipt: dict[str, object],
    receipt_sha: str,
    exception_id: str | None = None,
    registry_digest: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Require a distinct identified review of the exact receipt and custody.

    Parameters
    ----------
    repository : Path
        Resolved checkout containing the immutable review.
    entry : mapping of str to object
        Trusted index entry with the review path and digest.
    receipt : dict of str to object
        Actual receipt whose logs, sources and producer are reviewed.
    receipt_sha : str
        Digest of the exact receipt bytes.
    exception_id : str or None, optional
        Exact independently admitted historical execution exception.
    registry_digest : str or None, optional
        Digest of the registry that admitted that one exception.

    Returns
    -------
    tuple of Path and dict of str to object
        Confined review path and verified review fields.

    Raises
    ------
    ValueError
        If the reviewer, receipt, sources, logs or configuration do not bind.
    OSError
        If the required review file is unavailable.
    """
    log_entry = object_fields(receipt.get("log"))
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
        "configuration_sha256": (
            object_fields(entry["configuration"]).get("sha256")
            if "configuration" in entry
            else None
        ),
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
    return review_path, review
