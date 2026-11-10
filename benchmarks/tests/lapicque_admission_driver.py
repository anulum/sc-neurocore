# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reject real loader and loaded-source admission faults

"""Damage only process-local admission metadata and require refusal before measurement."""

from __future__ import annotations

from collections.abc import Callable
import importlib.machinery
from pathlib import Path
import runpy
import sys
from types import FrameType
from typing import cast


def main() -> None:
    """Exercise the public API with genuine loader state or corrupted loaded metadata."""
    source_arg, checkout_arg, output_arg, mode = sys.argv[1:]
    source, checkout, output = Path(source_arg), Path(checkout_arg), Path(output_arg)
    namespace = runpy.run_path(str(source), run_name="lapicque_admission_capture")
    entrypoint = cast(Callable[[list[str]], int], namespace["main"])
    harness = str((checkout / "benchmarks/bench_model_lapicque.py").resolve())
    previous_profile = sys.getprofile()
    previous_suffixes = list(importlib.machinery.SOURCE_SUFFIXES)

    def observe(frame: FrameType, event: str, result: object) -> None:
        """Corrupt real loaded admission metadata before any original main invocation."""
        if (
            frame.f_code.co_filename != harness
            or frame.f_code.co_name != "<module>"
            or event != "return"
        ):
            return
        if mode == "steps":
            frame.f_globals["N_STEPS"] += 1
        elif mode == "repeats":
            frame.f_globals["N_REPEATS"] += 1
        elif mode == "backends":
            frame.f_globals["BACKENDS"] = frame.f_globals["BACKENDS"][:-1]
        elif mode == "measurement-type":
            frame.f_globals["_measure_backend"] = None
        elif mode == "entrypoint-type":
            frame.f_globals["main"] = None
        elif mode == "measurement-origin":
            frame.f_globals["_measure_backend"] = entrypoint
        else:
            raise ValueError("unknown loaded-source admission fault")

    try:
        if mode == "loader":
            importlib.machinery.SOURCE_SUFFIXES.clear()
        else:
            sys.setprofile(observe)
        raise SystemExit(
            entrypoint(["--checkout", str(checkout), "--output-dir", str(output), "--diagnostic"])
        )
    finally:
        importlib.machinery.SOURCE_SUFFIXES[:] = previous_suffixes
        sys.setprofile(previous_profile)


if __name__ == "__main__":
    main()
