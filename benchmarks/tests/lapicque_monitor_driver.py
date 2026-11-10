# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real monitor-state changes after full original benchmark work

"""Change actual monitoring state after the original command finishes all measurements."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import CodeType, FrameType

from lapicque_cli_driver import main as run_original_cli


def main() -> None:
    """Exercise monitoring cleanup and preserve the actual state before fixture cleanup."""
    _, checkout_arg, output_arg, mode, _ = sys.argv[1:]
    harness = str((Path(checkout_arg) / "benchmarks/bench_model_lapicque.py").resolve())
    output = Path(output_arg)
    measured_code: CodeType | None = None
    changed_after_full_command = False
    foreign_name = "lapicque-monitor-ownership-test"
    previous_profile = sys.getprofile()
    stale_before_command = mode in {"stale-global", "stale-callback"}
    audit_interference = False

    def foreign_callback(*event: object) -> None:
        """Own a real replacement or stale monitor callback."""

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        """Introduce an actual owner loss during the real cleanup API call."""
        nonlocal audit_interference
        if (
            event == "sys.monitoring.register_callback"
            and not audit_interference
            and any(value is None for value in arguments)
            and sys.monitoring.get_tool(5) == "lapicque-raw-duration-return-observer"
        ):
            audit_interference = True
            sys.monitoring.free_tool_id(5)

    def observe(frame: FrameType, event: str, result: object) -> None:
        """Alter only monitoring metadata after all original compute and report writes."""
        nonlocal measured_code, changed_after_full_command
        if frame.f_code.co_filename != harness:
            return
        if frame.f_code.co_name == "_measure_backend":
            measured_code = frame.f_code
        if frame.f_code.co_name != "main" or event != "return" or result != 0:
            return
        assert measured_code is not None
        if mode == "global-events":
            sys.monitoring.set_events(5, sys.monitoring.events.LINE)
        elif mode == "local-events":
            sys.monitoring.set_local_events(5, measured_code, 0)
        elif mode == "callback-owner":
            sys.monitoring.register_callback(5, sys.monitoring.events.PY_RETURN, foreign_callback)
        elif mode == "cleanup-owner-loss":
            pass
        elif mode == "foreign-owner":
            sys.monitoring.set_local_events(5, measured_code, 0)
            sys.monitoring.register_callback(5, sys.monitoring.events.PY_RETURN, None)
            sys.monitoring.free_tool_id(5)
            sys.monitoring.use_tool_id(5, foreign_name)
        else:
            raise ValueError("unknown monitoring scenario")
        changed_after_full_command = True

    if stale_before_command:
        sys.monitoring.use_tool_id(5, foreign_name)
        if mode == "stale-global":
            sys.monitoring.set_events(5, sys.monitoring.events.LINE)
        else:
            sys.monitoring.register_callback(5, sys.monitoring.events.PY_RETURN, foreign_callback)
        sys.monitoring.free_tool_id(5)
    if mode == "cleanup-owner-loss":
        sys.addaudithook(audit)
    sys.setprofile(observe)
    try:
        run_original_cli()
    finally:
        owner = sys.monitoring.get_tool(5)
        state = dict(
            scenario=mode,
            stale_before_command=stale_before_command,
            audit_interference=audit_interference,
            changed_after_full_command=changed_after_full_command,
            owner_after_command=owner,
            global_events_after_command=sys.monitoring.get_events(5),
            profile_owner_preserved=sys.getprofile() is observe,
            measured_source_filename=None if measured_code is None else measured_code.co_filename,
        )
        sys.setprofile(previous_profile)
        if (changed_after_full_command or stale_before_command) and (
            owner is None or owner == foreign_name
        ):
            if owner is None:
                sys.monitoring.use_tool_id(5, foreign_name)
            sys.monitoring.set_events(5, 0)
            if measured_code is not None:
                sys.monitoring.set_local_events(5, measured_code, 0)
            sys.monitoring.register_callback(5, sys.monitoring.events.PY_RETURN, None)
            sys.monitoring.free_tool_id(5)
        with (output.parent / "actual-monitor-state.json").open("x") as stream:
            json.dump(state, stream)


if __name__ == "__main__":
    main()
