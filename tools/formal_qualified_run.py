# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Primary proof and assertion-enable qualification

"""Run a native proof and its enable audit inside one caller-supplied budget."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.formal_evidence_validation import digest, require
from tools.formal_job_inputs import resolve_job
from tools.formal_nonvacuity import check_task


def main(argv: Sequence[str] | None = None) -> int:
    """Run every default task, then audit each successful prepared model.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments. The existing external timeout and cgroup must wrap
        this entire command, including auxiliary model generation and covers.

    Returns
    -------
    int
        Zero only when both the native primary and all enable audits pass;
        one for a native counterexample and two for missing qualification.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "ERROR",
        "primary_status": "NOT_RUN",
        "nonvacuity_status": "NOT_RUN",
        "tasks": {},
    }
    exit_code = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        inputs = resolve_job(args.repository, args.job)
        root = args.work_root.resolve()
        require(not root.exists(), "Proof output already exists; refusing a replay.")
        root.mkdir(parents=True)
        report["source_hashes"] = inputs.source_hashes()
        defaults = subprocess.run(
            ["sby", "--dumpdefaults", str(inputs.job)],
            cwd=inputs.directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines() or [""]
        require(len(set(defaults)) == len(defaults), "Native default tasks are ambiguous.")
        with (root / "primary.log").open("w") as stream:
            primary = subprocess.run(
                ["sby", "--prefix", str(root / "proof"), str(inputs.job)],
                cwd=inputs.directory,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        report["primary_exit_code"] = primary.returncode
        native_log = args.report.with_suffix(".native.log")
        shutil.copyfile(root / "primary.log", native_log)
        report["primary_log_sha256"] = digest(native_log)
        task_roots = {name: root / ("proof_" + name if name else "proof") for name in defaults}
        statuses = {
            name: (directory / "status").read_text().split()[0]
            if (directory / "status").is_file()
            else "ERROR"
            for name, directory in task_roots.items()
        }
        report["primary_task_statuses"] = statuses
        native_pass = primary.returncode == 0 and all(s == "PASS" for s in statuses.values())
        report["primary_status"] = (
            "PASS" if native_pass else ("FAIL" if "FAIL" in statuses.values() else "ERROR")
        )
        if native_pass:
            report["nonvacuity_status"] = "RUNNING"
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            tasks = {}
            for index, (name, directory) in enumerate(task_roots.items()):
                tasks[name] = check_task(directory, root / f"nonvacuity_{index}")
                auxiliary_log = root / f"nonvacuity_{index}/cover.log"
                if auxiliary_log.is_file():
                    shutil.copyfile(auxiliary_log, args.report.with_suffix(f".enable_{index}.log"))
                report["tasks"] = tasks
                args.report.write_text(json.dumps(report, indent=2) + "\n")
            qualified = all(task["verdict"] == "NONVACUOUS" for task in tasks.values())
            report["nonvacuity_status"] = "NONVACUOUS" if qualified else "ERROR"
            report["status"] = "PASS" if qualified else "ERROR"
            exit_code = 0 if qualified else 2
        elif report["primary_status"] == "FAIL":
            report["status"] = "FAIL"
            exit_code = 1
        require(
            inputs.source_hashes() == report["source_hashes"],
            "Proof sources changed during execution.",
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report.update(status="ERROR", error=str(exc))
        if report["primary_status"] == "PASS":
            report["nonvacuity_status"] = "ERROR"
        exit_code = 2
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
