# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real post-measurement source and output integrity faults


"""Change real source/output conditions only after complete original measurements."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import FrameType

from lapicque_cli_driver import main as run_original_cli


def main() -> None:
    """Exercise real integrity failures without changing clocks or numerical computation."""
    _, checkout_arg, output_arg, mode, _ = sys.argv[1:]
    checkout, output = Path(checkout_arg), Path(output_arg)
    harness = (checkout / "benchmarks/bench_model_lapicque.py").resolve()
    report = output / "original-harness-report.json"
    preserved_report = output / "preserved-original-report.json"
    events: list[str] = []
    changed_after_full_command = False
    read_fault_triggered = False
    previous_profile = sys.getprofile()

    def observe(frame: FrameType, event: str, result: object) -> None:
        """Observe actual returns and alter only the intended post-computation condition."""
        nonlocal changed_after_full_command
        if frame.f_code.co_filename != str(harness):
            return
        if frame.f_code.co_name == "_measure_backend" and event == "return":
            events.append(str(frame.f_locals["backend"]))
        if frame.f_code.co_name != "main" or event != "return" or result != 0:
            return
        if mode == "source-removed":
            assert checkout.parent == output.parent and not harness.is_symlink()
            harness.unlink()
        elif mode == "backend-metadata":
            assert frame.f_back is not None
            state = frame.f_back.f_locals["capture"]
            rows = state.rows
            assert isinstance(rows, list) and len(rows) == 5 and isinstance(rows[0], dict)
            rows[0]["backend"] = False
        elif mode == "retention-only-collision":
            (output / "original-seven-duration-samples.json").write_text("preserved sentinel")
        elif mode != "report-read-failure":
            raise ValueError("unknown post-computation integrity fault")
        changed_after_full_command = True

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        """Cause a genuine read failure after the report's existence check, for any UID."""
        nonlocal read_fault_triggered
        if (
            mode == "report-read-failure"
            and changed_after_full_command
            and not read_fault_triggered
            and event == "open"
            and arguments
            and arguments[0] == str(report)
        ):
            read_fault_triggered = True
            report.rename(preserved_report)
            report.mkdir()

    if mode == "report-read-failure":
        sys.addaudithook(audit)
    sys.setprofile(observe)
    try:
        run_original_cli()
    finally:
        profile_preserved = sys.getprofile() is observe
        sys.setprofile(previous_profile)
        if read_fault_triggered:
            report.rmdir()
            preserved_report.rename(report)
        with (output.parent / "actual-integrity-state.json").open("x") as stream:
            json.dump(
                dict(
                    mode=mode,
                    actual_backend_returns=events,
                    changed_after_full_command=changed_after_full_command,
                    actual_report_read_fault=read_fault_triggered,
                    profile_owner_preserved=profile_preserved,
                    monitoring_owner_after_cli=sys.monitoring.get_tool(5),
                ),
                stream,
            )


if __name__ == "__main__":
    main()
