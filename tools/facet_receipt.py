#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Record an immutable facet receipt by running its evidence command

r"""Record a facet receipt by executing the declared evidence command.

The recorder derives the facet's subjects from the registry (descriptor
contract, model module, schema profiles, source reference, compiler tree,
committed RTL, report and validator files), runs the command, captures the
tool, runtime and outcome, seals the receipt and writes it as a new file under
``src/sc_neurocore/neurons/facet_receipts``. It never overwrites an existing
receipt: a later run is a new file that supersedes the older one.

A pytest command is run with a JUnit report so the passed/failed/skipped
counts and executed validator identities are retained from the run itself.
Other commands cannot credit a facet until a result adapter is implemented.

Usage::

    python tools/facet_receipt.py record --model LapicqueNeuron --facet cosim --profile lapicque \\
        -- python -m pytest "tests/test_cosim_lapicque.py::test_source_q3232_preserves_first_attainment_and_polarization_bound"
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore import __version__ as PACKAGE_VERSION  # noqa: E402
from sc_neurocore.neurons.evidence_references import (  # noqa: E402
    parse_evidence_field,
    sha256_file,
)
from sc_neurocore.neurons.facet_receipts import (  # noqa: E402
    FACET_BY_NAME,
    FACET_RECEIPT_SCHEMA,
    RECEIPT_DIR,
    SUBJECT_KINDS,
    FacetReceipt,
    FacetReceiptError,
    Subject,
    credit_problems,
    receipt_filename,
)
from sc_neurocore.neurons.model_identity import identity_registry  # noqa: E402
from sc_neurocore.neurons.readiness import (  # noqa: E402
    current_digest,
    derive_subjects,
    facet_evidence_field,
)
from sc_neurocore.neurons.receipt_execution import (  # noqa: E402
    EXECUTION_CONTRACT,
    evidence_selection_problems,
    execution_problems,
    junit_checks,
    pytest_command,
)

_SILICON_TOOLS = ("iverilog", "yosys", "verilator", "sby")
_RTL_DIR = "hdl/formal/catalogue"


def _formal_inventory_tool() -> ModuleType:
    """Load ``tools/emit_catalogue_formal.py`` for its class-to-RTL mapping."""
    path = REPO_ROOT / "tools" / "emit_catalogue_formal.py"
    spec = importlib.util.spec_from_file_location("emit_catalogue_formal_mapping", path)
    if spec is None or spec.loader is None:
        raise RecordError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # slotted dataclasses resolve their module here
    spec.loader.exec_module(module)
    return module


def committed_rtl_path(class_name: str) -> str | None:
    """Return the committed formal-lane RTL of ``class_name``, if one is tracked.

    The module name follows ``tools/emit_catalogue_formal.py``: a curated
    module name when the class has one, otherwise the schema's generated module
    name; the path is returned only when the file exists.
    """
    tool = _formal_inventory_tool()
    schema = (
        tool.CLASS_TO_SCHEMA.get(class_name)
        or tool.CURATED_CLASS_TO_SCHEMA.get(class_name)
        or tool.RETAINED_SC_CLASS_TO_SCHEMA.get(class_name)
    )
    module = tool.CURATED_CLASS_TO_MODULE.get(class_name)
    if module is None and schema is not None:
        module = tool.MODULE_NAME_BY_SCHEMA.get(schema, f"sc_{schema}")
    if module is None:
        return None
    relative = f"{_RTL_DIR}/{module}.v"
    return relative if (REPO_ROOT / relative).is_file() else None


class RecordError(RuntimeError):
    """Raised when a receipt cannot be recorded as requested."""


def _tool_version(name: str) -> str:
    """Return the first line a tool prints for ``-V``/``--version``, or empty."""
    executable = shutil.which(name)
    if executable is None:
        return ""
    for flag in ("-V", "--version"):
        try:
            completed = subprocess.run(
                [executable, flag], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (completed.stdout or completed.stderr).strip().splitlines()
        if text:
            return text[0][:200]
    return ""


def _git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _runtime(subjects: tuple[Subject, ...]) -> dict[str, str]:
    dirty = _git(["status", "--porcelain", "--", *(subject.path for subject in subjects)])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sc_neurocore": PACKAGE_VERSION,
        "git_head": _git(["rev-parse", "HEAD"]),
        "dirty_subjects": ", ".join(sorted(line[3:] for line in dirty.splitlines())),
    }


def _junit_counts(report: Path) -> dict[str, int]:
    return junit_checks(report.read_text(encoding="utf-8"))[0]


def _outcome(exit_code: int, counts: dict[str, int]) -> str:
    if counts["failed"] > 0:
        return "failed"
    if counts["errors"] > 0 or exit_code != 0:
        return "error"
    if counts["passed"] < 1:
        return "skipped"
    return "passed"


def run_command(
    command: list[str], *, timeout: float | None
) -> tuple[int, dict[str, int], str, dict[str, str]]:
    """Run the evidence command and return exit code, counts, outcome and tool.

    Direct pytest invocations receive ``-p no:cacheprovider --junitxml`` and
    counts come from testcase elements. Other successful commands carry zero
    passed checks. None means a bounded 600-second budget, not unlimited work.
    """
    duration = 600.0 if timeout is None else timeout
    if not math.isfinite(duration) or duration <= 0:
        raise RecordError("timeout must be positive and finite")
    is_pytest = pytest_command(command)
    tool = {"name": "pytest" if is_pytest else Path(command[0]).name}
    if is_pytest:
        import pytest

        tool["version"] = pytest.__version__
    else:
        tool["version"] = _tool_version(command[0])
    with tempfile.TemporaryDirectory(prefix="facet-receipt-") as scratch:
        argv = list(command)
        report = Path(scratch) / "junit.xml"
        if is_pytest:
            argv += ["-p", "no:cacheprovider", f"--junitxml={report}"]
        try:
            process = subprocess.Popen(
                argv,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
            )
            stdout, stderr = process.communicate(timeout=duration)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate()
            counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
            return -1, counts, "timeout", tool
        exit_code = process.returncode
        tool["stdout"] = stdout
        tool["stderr"] = stderr
        if is_pytest and report.is_file():
            tool["junit_xml"] = report.read_text(encoding="utf-8")
            try:
                counts = _junit_counts(report)
            except ValueError:
                counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0}
        elif is_pytest:
            counts = {"collected": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0}
        else:
            counts = {
                "collected": 1,
                "passed": 0,
                "failed": 0 if exit_code == 0 else 1,
                "errors": 0,
                "skipped": 0,
            }
    return exit_code, counts, _outcome(exit_code, counts), tool


def _parse_extra_subject(text: str) -> Subject:
    kind, separator, path = text.partition("=")
    if not separator or not kind or not path:
        raise RecordError(f"--subject expects KIND=PATH, got {text!r}")
    if kind not in SUBJECT_KINDS:
        raise RecordError(f"--subject kind {kind!r} is not one of {', '.join(SUBJECT_KINDS)}")
    target = REPO_ROOT / path
    if not target.is_file():
        raise RecordError(f"--subject path does not exist: {path}")
    return Subject(kind, path, sha256_file(target))


def record_receipt(
    *,
    model: str,
    facet: str,
    command: list[str],
    profile: str = "",
    claim_scope: str | None = None,
    evidence: list[str] | None = None,
    extra_subjects: list[Subject] | None = None,
    notes: str = "",
    timeout: float | None = None,
    receipt_dir: Path = RECEIPT_DIR,
    recorded_at: str | None = None,
) -> tuple[Path, FacetReceipt]:
    """Run ``command`` for ``model``/``facet`` and write a sealed receipt.

    Raises
    ------
    RecordError
        If the model or facet is unknown, no evidence reference can be found,
        the derived subjects miss a required kind, or the target file exists.
    """
    spec = FACET_BY_NAME.get(facet)
    if spec is None:
        raise RecordError(f"unknown facet {facet!r}; known: {', '.join(FACET_BY_NAME)}")
    registry = identity_registry()
    if model not in registry or registry[model].kind == "api-alias":
        raise RecordError(f"{model!r} is not a registered class")
    identity = registry[model]
    profiles = {item.stem for item in identity.schema_profiles} or {"hand"}
    if not profile:
        if len(profiles) != 1:
            raise RecordError("multiple model profiles: select --profile explicitly")
        profile = next(iter(profiles))
    if profile not in profiles:
        raise RecordError(f"unknown profile {profile!r} for {model}")
    refs = list(evidence or ())
    from sc_neurocore.neurons.model_catalogue import load_descriptor

    descriptor = load_descriptor(model)
    field_text = ""
    if descriptor is not None:
        field_text = facet_evidence_field(descriptor, spec)
        if field_text and not refs:
            refs.append(field_text)
    if not refs:
        raise RecordError(f"{model} declares no evidence reference for {facet}; pass --evidence")
    for raw in refs:
        for reference in parse_evidence_field(raw, REPO_ROOT):
            if reference.is_locatable and not reference.is_resolved:
                raise RecordError(f"evidence reference does not resolve: {reference.raw}")
    selection_errors = evidence_selection_problems(refs, field_text)
    if selection_errors:
        raise RecordError("; ".join(selection_errors))
    command_files = [
        arg.split("::", 1)[0]
        for arg in command
        if not arg.startswith(("/", "-"))
        and arg.split("::", 1)[0].endswith(".py")
        and arg.split("::", 1)[0].startswith("tests/")
        and (REPO_ROOT / arg.split("::", 1)[0]).is_file()
    ]
    extras = list(extra_subjects or ())
    if "committed-rtl" in spec.subjects and not any(s.kind == "committed-rtl" for s in extras):
        rtl = committed_rtl_path(model)
        if rtl is not None:
            extras.append(Subject("committed-rtl", rtl, sha256_file(REPO_ROOT / rtl)))
    subjects = derive_subjects(
        model,
        facet,
        repo_root=REPO_ROOT,
        evidence_refs=[*refs, *command_files],
        extra_subjects=extras,
    )
    missing = [kind for kind in spec.required_subjects if kind not in {s.kind for s in subjects}]
    if missing:
        raise RecordError(
            f"cannot derive required subject kind(s) {', '.join(missing)} for {model}/{facet}; "
            "pass them with --subject KIND=PATH"
        )
    if not pytest_command(command):
        raise RecordError(
            "credit requires a direct pytest validator; native result adapters are not yet supported"
        )
    exit_code, counts, outcome, tool = run_command(command, timeout=timeout)
    validator = {
        "name": "tools/facet_receipt.py",
        "schema": FACET_RECEIPT_SCHEMA,
        "execution_contract": EXECUTION_CONTRACT,
        "junit_xml": tool.pop("junit_xml", ""),
    }
    execution_errors = execution_problems(command, refs, validator, counts)
    changed = [
        subject.path for subject in subjects if current_digest(subject, REPO_ROOT) != subject.sha256
    ]
    after = derive_subjects(
        model,
        facet,
        repo_root=REPO_ROOT,
        evidence_refs=[*refs, *command_files],
        extra_subjects=extras,
    )
    if {(s.kind, s.path, s.scope) for s in after} != {(s.kind, s.path, s.scope) for s in subjects}:
        changed.append("subject manifest membership")
    if execution_errors or changed:
        outcome = "error"
        notes = "\n".join(
            [notes, *execution_errors, *(f"subject changed during run: {p}" for p in changed)]
        )
    extra_tools = (
        {name: _tool_version(name) for name in _SILICON_TOOLS if shutil.which(name)}
        if spec.axis == "silicon"
        else {}
    )
    stamp = recorded_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = FacetReceipt(
        class_name=model,
        facet=facet,
        profile=profile,
        claim_scope=spec.claim_scope if claim_scope is None else claim_scope,
        subjects=subjects,
        evidence_refs=tuple(refs),
        command=tuple(command),
        tool=tool,
        extra_tools=extra_tools,
        runtime=_runtime(subjects),
        validator=validator,
        outcome=outcome,
        exit_code=exit_code,
        counts=counts,
        recorded_at=stamp,
        notes=notes,
    ).sealed()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    target = receipt_dir / receipt_filename(model, facet, stamp, profile=profile)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=receipt_dir, delete=False
    ) as pending:
        temporary = Path(pending.name)
        try:
            pending.write(json.dumps(receipt.to_payload(), indent=2, ensure_ascii=False) + "\n")
            pending.flush()
            os.fsync(pending.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise RecordError(f"receipt already exists and is immutable: {target}") from error
        finally:
            temporary.unlink()
    return target, receipt


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="action", required=True)
    record = commands.add_parser("record", help="run the evidence command and write a receipt")
    record.add_argument("--model", required=True, help="registered class name")
    record.add_argument("--facet", required=True, help="facet name, e.g. cosim or backend:rust")
    record.add_argument("--profile", default="", help="schema profile stem or 'hand'")
    record.add_argument("--claim-scope", default=None, help="override the facet's claim scope")
    record.add_argument("--evidence", action="append", default=[], help="evidence reference")
    record.add_argument("--subject", action="append", default=[], help="extra subject KIND=PATH")
    record.add_argument("--notes", default="", help="free-text note stored in the receipt")
    record.add_argument("--timeout", type=float, default=None, help="command timeout in seconds")
    record.add_argument("--receipt-dir", type=Path, default=RECEIPT_DIR)
    record.add_argument("command", nargs=argparse.REMAINDER, help="evidence command after --")
    args = parser.parse_args(argv)
    command = [arg for arg in args.command if arg != "--"]
    if not command:
        parser.error("an evidence command is required after --")
    try:
        extra = [_parse_extra_subject(item) for item in args.subject]
        path, receipt = record_receipt(
            model=args.model,
            facet=args.facet,
            command=command,
            profile=args.profile,
            claim_scope=args.claim_scope,
            evidence=args.evidence,
            extra_subjects=extra,
            notes=args.notes,
            timeout=args.timeout,
            receipt_dir=args.receipt_dir,
        )
    except (RecordError, FacetReceiptError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    problems = credit_problems(receipt, class_name=args.model)
    print(f"wrote {path}: outcome={receipt.outcome}")
    if problems:
        for problem in problems:
            print(f"  not creditable: {problem}")
        return 1
    print(f"  creditable for {args.facet} ({len(receipt.subjects)} subjects sealed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
