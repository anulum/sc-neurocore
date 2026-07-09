# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Icarus Verilog co-simulation dependency check

"""Verify the Icarus Verilog co-simulation toolchain is available."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Sequence
from typing import Callable

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

_IVERILOG_VERSION_RE = re.compile(r"Icarus Verilog version\s+(\d+)(?:\.(\d+))?")


def parse_iverilog_major(version_output: str) -> int | None:
    """Return the Icarus Verilog major version from ``iverilog -V`` output."""
    match = _IVERILOG_VERSION_RE.search(version_output)
    if match is None:
        return None
    return int(match.group(1))


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a version command and capture its text output."""
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    """Return stdout and stderr text from a completed version command."""
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def check_icarus_verilog(
    *,
    minimum_major: int,
    runner: CommandRunner = _run_command,
) -> list[str]:
    """Return dependency errors for the local Icarus Verilog toolchain."""
    errors: list[str] = []

    try:
        iverilog_result = runner(("iverilog", "-V"))
    except FileNotFoundError:
        return ["iverilog executable is not available on PATH"]

    iverilog_output = _combined_output(iverilog_result)
    if iverilog_result.returncode != 0:
        errors.append(f"iverilog -V failed with exit code {iverilog_result.returncode}")
    major = parse_iverilog_major(iverilog_output)
    if major is None:
        errors.append("iverilog -V output did not include an Icarus Verilog version")
    elif major < minimum_major:
        errors.append(
            f"Icarus Verilog {major}.x is below the required {minimum_major}.x floor"
        )

    try:
        vvp_result = runner(("vvp", "-V"))
    except FileNotFoundError:
        errors.append("vvp executable is not available on PATH")
    else:
        if vvp_result.returncode != 0:
            errors.append(f"vvp -V failed with exit code {vvp_result.returncode}")

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line co-simulation dependency check."""
    parser = argparse.ArgumentParser(
        description="Verify Icarus Verilog and vvp for SC-NeuroCore co-simulation."
    )
    parser.add_argument(
        "--minimum-major",
        type=int,
        default=12,
        help="Minimum supported Icarus Verilog major version.",
    )
    args = parser.parse_args(argv)

    errors = check_icarus_verilog(minimum_major=args.minimum_major)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[OK] Icarus Verilog toolchain satisfies {args.minimum_major}.x floor")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point.
    raise SystemExit(main())
