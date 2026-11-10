# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real Lapicque command execution and diagnostic file failures

"""Run unchanged benchmark work and apply real filesystem failures at its boundary."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import runpy
import sys
from types import FrameType
from typing import cast


def main() -> None:
    """Execute the public CLI or public API with genuine diagnostic conditions."""
    source_arg, checkout_arg, output_arg, mode, status_arg = sys.argv[1:]
    source, checkout, output = Path(source_arg), Path(checkout_arg), Path(output_arg)
    harness = str((checkout / "benchmarks/bench_model_lapicque.py").resolve())
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    events: list[dict[str, object]] = []
    foreign_slot = mode == "occupied-slot"
    profiled = mode in {
        "diagnostic",
        "corrupt-report",
        "partial-report",
        "harness-io-failure",
        "retention-collision",
        "refuse-profile",
    }

    def observe(frame: FrameType, event: str, value: object) -> None:
        """Observe real returns and alter only actual output filesystem conditions."""
        if frame.f_code.co_filename != harness:
            return
        if frame.f_code.co_name == "_measure_backend" and event == "return":
            events.append(
                dict(event="actual_measurement_return", backend=frame.f_locals["backend"])
            )
        if frame.f_code.co_name != "main":
            return
        if event == "call" and mode in {"harness-io-failure", "retention-collision"}:
            (output / "original-harness-report.json").mkdir()
            if mode == "retention-collision":
                (output / "original-seven-duration-samples.json").write_text("preserved sentinel")
        if event == "return" and value == 0 and mode in {"corrupt-report", "partial-report"}:
            report = output / "original-harness-report.json"
            (output / "unmodified-harness-report.json").write_bytes(report.read_bytes())
            report.write_text("{" if mode == "corrupt-report" else '{"backends": {}}')

    previous_profile = sys.getprofile()
    if foreign_slot:
        sys.monitoring.use_tool_id(5, "lapicque-cli-existing-slot-test")
    if profiled:
        sys.setprofile(observe)
    arguments = ["--checkout", str(checkout), "--output-dir", str(output)]
    if mode not in {"normal", "outer-exception", "refuse-profile"}:
        arguments.append("--diagnostic")
    sys.argv = [str(source), *arguments]
    try:
        if mode in {"outer-exception", "outer-diagnostic"}:
            try:
                raise LookupError("caller exception already handled")
            except LookupError:
                runpy.run_path(str(source), run_name="__main__")
        elif mode == "api-diagnostic":
            namespace = runpy.run_path(str(source), run_name="lapicque_capture_api")
            entrypoint = cast(Callable[[list[str]], int], namespace["main"])
            raise SystemExit(entrypoint(arguments))
        else:
            runpy.run_path(str(source), run_name="__main__")
    finally:
        slot = sys.monitoring.get_tool(5)
        status = dict(
            actual_harness_events=events,
            profile_callback_preserved=not profiled or sys.getprofile() is observe,
            slot_after_cli=slot,
            foreign_slot_preserved=not foreign_slot or slot == "lapicque-cli-existing-slot-test",
        )
        if profiled:
            sys.setprofile(previous_profile)
        if foreign_slot and slot == "lapicque-cli-existing-slot-test":
            sys.monitoring.free_tool_id(5)
        with Path(status_arg).open("x") as stream:
            json.dump(status, stream)


if __name__ == "__main__":
    main()
