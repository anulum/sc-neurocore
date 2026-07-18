# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EDA toolchain version inventory helper

"""Collect EDA toolchain versions for hardware evidence manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sc-neurocore.eda-toolchain.v1"


@dataclass(frozen=True)
class ToolSpec:
    """One external tool to probe."""

    key: str
    label: str
    commands: tuple[tuple[str, ...], ...]
    accepted_returncodes: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class Finding:
    """One inventory expectation failure."""

    level: str
    message: str


Runner = Callable[..., subprocess.CompletedProcess[str]]
VersionLookup = Callable[[str], str]


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("vivado", "AMD Vivado", (("vivado", "-version"),)),
    ToolSpec("openroad", "OpenROAD", (("openroad", "-version"), ("openroad", "--version"))),
    ToolSpec("yosys", "Yosys", (("yosys", "--version"), ("yosys", "-V"))),
    ToolSpec("nextpnr_ice40", "nextpnr iCE40", (("nextpnr-ice40", "--version"),)),
    ToolSpec("nextpnr_ecp5", "nextpnr ECP5", (("nextpnr-ecp5", "--version"),)),
    # icepack has no version flag. Its help/usage path exits 1 even when the
    # executable is healthy, so treat that documented probe result as presence.
    ToolSpec("icepack", "Project IceStorm icepack", (("icepack", "--help"),), (0, 1)),
    ToolSpec("ecppack", "Project Trellis ecppack", (("ecppack", "--version"),)),
    ToolSpec("quartus", "Intel Quartus", (("quartus_sh", "--version"),)),
    ToolSpec("lattice_diamond", "Lattice Diamond", (("diamondc", "--version"),)),
    ToolSpec("lattice_radiant", "Lattice Radiant", (("radiantc", "--version"),)),
)


def collect_eda_toolchain_versions(
    *,
    runner: Runner = subprocess.run,
    version_lookup: VersionLookup = metadata.version,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable EDA toolchain inventory."""
    env = os.environ if environ is None else environ
    tools = {spec.key: _probe_tool(spec, runner=runner) for spec in TOOL_SPECS}
    tools["pynq"] = _probe_python_distribution("pynq", "PYNQ", version_lookup)

    return {
        "schema_version": SCHEMA_VERSION,
        "tools": tools,
        "environment": {
            "openroad_image_digest": env.get("OPENROAD_IMAGE_DIGEST"),
            "pdk": env.get("PDK"),
            "pdk_root_set": bool(env.get("PDK_ROOT")),
        },
    }


def check_expectations(
    report: dict[str, Any],
    *,
    required: Sequence[str] = (),
    expected_versions: Sequence[str] = (),
) -> list[Finding]:
    """Validate required tools and version-substring expectations."""
    findings: list[Finding] = []
    tools = report.get("tools", {})
    if not isinstance(tools, dict):
        return [Finding("error", "report has no tools mapping")]

    for key in required:
        tool = tools.get(key)
        if not isinstance(tool, dict):
            findings.append(Finding("error", f"required tool {key!r} is not in the inventory"))
        elif not tool.get("available"):
            findings.append(Finding("error", f"required tool {key!r} is not available"))

    for expectation in expected_versions:
        key, expected = _parse_expectation(expectation)
        tool = tools.get(key)
        version = tool.get("version") if isinstance(tool, dict) else None
        if not isinstance(version, str) or expected not in version:
            findings.append(
                Finding(
                    "error",
                    f"tool {key!r} version does not contain {expected!r}; found {version!r}",
                )
            )

    return findings


def _probe_tool(spec: ToolSpec, *, runner: Runner) -> dict[str, Any]:
    last_error: str | None = None
    for command in spec.commands:
        try:
            result = runner(
                list(command),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            last_error = "not found"
            continue
        except subprocess.TimeoutExpired:
            return _tool_record(spec, command, False, None, "timeout")

        output = _normalise_output(result.stdout, result.stderr)
        if result.returncode in spec.accepted_returncodes:
            return _tool_record(spec, command, True, output, None)
        last_error = output or f"exit {result.returncode}"

    return _tool_record(spec, spec.commands[0], False, None, last_error)


def _probe_python_distribution(
    distribution: str,
    label: str,
    version_lookup: VersionLookup,
) -> dict[str, Any]:
    try:
        version = version_lookup(distribution)
    except metadata.PackageNotFoundError:
        return {
            "label": label,
            "available": False,
            "version": None,
            "command": None,
            "executable": None,
            "error": "not installed",
        }
    return {
        "label": label,
        "available": True,
        "version": version,
        "command": f"python -m importlib.metadata {distribution}",
        "executable": "python",
        "error": None,
    }


def _tool_record(
    spec: ToolSpec,
    command: Sequence[str],
    available: bool,
    version: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "label": spec.label,
        "available": available,
        "version": version,
        "command": " ".join(command),
        "executable": command[0],
        "error": error,
    }


def _normalise_output(stdout: str, stderr: str) -> str | None:
    for text in (stdout, stderr):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _parse_expectation(expectation: str) -> tuple[str, str]:
    key, separator, expected = expectation.partition("=")
    if not separator or not key or not expected:
        raise argparse.ArgumentTypeError("--expect entries must use TOOL=VERSION_SUBSTRING syntax")
    return key, expected


def main(argv: Sequence[str] | None = None, *, runner: Runner = subprocess.run) -> int:
    """Run the EDA version inventory CLI."""
    parser = argparse.ArgumentParser(description="Collect SC-NeuroCore EDA tool versions")
    parser.add_argument("--out", type=Path, help="Write JSON inventory to this file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="TOOL",
        help="Fail if TOOL is missing or unavailable",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="TOOL=VERSION_SUBSTRING",
        help="Fail if TOOL version does not contain VERSION_SUBSTRING",
    )
    args = parser.parse_args(argv)

    try:
        report = collect_eda_toolchain_versions(runner=runner)
        findings = check_expectations(
            report,
            required=args.require,
            expected_versions=args.expect,
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    report["findings"] = [finding.__dict__ for finding in findings]
    report["passed"] = not findings

    payload = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)

    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
