# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Record the physical-GPU evidence behind the coverage platform exception

r"""Write the physical-GPU evidence record that the coverage exception gate checks.

The GPU waveform encoder runs only on a physical GPU adapter, so its success line is a
registered platform exception (``tools/coverage_platform_exceptions.json``). The owner's
decision backs that exception with evidence from a physical GPU. On such a host, three
measurements are taken with the public API, as ``docs/development/maintenance_tools.md``
describes:

- the adapter the encoder selects, written as ``{"name", "api", "engine_version"}``;
- a JUnit report of the GPU encoder test, which compares GPU packets with CPU packets and
  decodes them on a declared input;
- a ``coverage json`` report of that test run.

This tool reads the three files, refuses them unless the adapter is on a physical one's graphics
API and not named as a known software adapter, the test passed, and every registered line ran,
and writes the record with the sha256 of
each GPU source the register names. The coverage exception gate fails hosted CI when any of
those sources later differs from the record.

Usage::

    python tools/gpu_waveform_encoder_evidence.py --adapter adapter.json \
        --junit junit.xml --coverage-json coverage.json --python-version 3.12.16

Exit status: 0 when the record is written, 2 when an input refuses it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coverage_exception_gate import (
    EVIDENCE_SCHEMA,
    GateInputError,
    file_sha256,
    load_register,
)

ROOT = Path(__file__).resolve().parents[1]

#: The test whose passing run is the evidence: GPU packets equal CPU packets and decode back.
GPU_TEST = (
    "tests/test_waveform_stream.py::test_installed_gpu_encoder_uses_the_reported_physical_backend"
)

#: The declared input of that test, named so that the record says what was compared.
DECLARED_INPUT = (
    "_adc_stream(26, 3, 71) in tests/test_waveform_stream.py: windows of 12, 12, 2 and 0 "
    "samples, then 2 samples after reset"
)

#: The graphics APIs of a physical adapter the engine accepts.
PHYSICAL_APIS = ("Vulkan", "Metal", "Dx12")

#: Names of known software adapters, matched case-insensitively within the adapter's name.
SOFTWARE_ADAPTER_NAMES = ("llvmpipe", "lavapipe", "swiftshader", "warp", "microsoft basic render")


class EvidenceError(ValueError):
    """An input refuses the evidence record."""


def _test_outcome(junit_text: str) -> str:
    """Return the outcome of :data:`GPU_TEST` in a JUnit report: passed, failed or skipped."""
    module, name = GPU_TEST.split("::")
    classname = module.removesuffix(".py").replace("/", ".")
    try:
        tree = ElementTree.fromstring(junit_text)
    except ElementTree.ParseError as error:
        raise EvidenceError(f"the JUnit report cannot be parsed: {error}") from error
    for case in tree.iter("testcase"):
        if case.get("name") == name and case.get("classname") == classname:
            if case.find("skipped") is not None:
                return "skipped"
            if case.find("failure") is not None or case.find("error") is not None:
                return "failed"
            return "passed"
    raise EvidenceError(f"the JUnit report has no result for {GPU_TEST}")


def build_record(
    root: Path,
    adapter: dict[str, Any],
    junit_text: str,
    coverage_report: dict[str, Any],
    python_version: str,
    recorded_at: str,
) -> dict[str, Any]:
    """Return the evidence record for one physical-GPU run, or refuse the run.

    Parameters
    ----------
    root : Path
        The checkout.
    adapter : dict
        ``{"name", "api", "engine_version"}`` of the adapter the encoder selected.
    junit_text : str
        The JUnit report of the run of :data:`GPU_TEST`.
    coverage_report : dict
        The ``coverage json`` report of that run.
    python_version : str
        The interpreter that ran it.
    recorded_at : str
        The UTC time of the record.

    Returns
    -------
    dict
        The record, ready to be written as JSON.

    Raises
    ------
    EvidenceError
        If the adapter's graphics API is not a physical one's, its name is a known software
        adapter's, the test did not pass, or a registered line did not run.
    """
    try:
        register, entries = load_register(root)
    except GateInputError as error:
        raise EvidenceError(str(error)) from error
    name, api = adapter.get("name"), adapter.get("api")
    if not name or api not in PHYSICAL_APIS:
        raise EvidenceError(f"adapter {name!r} on {api!r} is not a physical adapter's graphics API")
    software = [known for known in SOFTWARE_ADAPTER_NAMES if known in str(name).lower()]
    if software:
        raise EvidenceError(f"adapter {name!r} is a known software adapter ({software[0]})")
    outcome = _test_outcome(junit_text)
    if outcome != "passed":
        raise EvidenceError(f"{GPU_TEST} {outcome}; the evidence needs it to pass on the GPU")
    files = coverage_report.get("files", {})
    covered = []
    for entry in entries:
        executed = files.get(entry.path, {}).get("executed_lines", [])
        if entry.line not in executed:
            raise EvidenceError(f"{entry.path}:{entry.line} did not run on the GPU")
        covered.append({"path": entry.path, "line": entry.line})
    digests = {}
    for path in register["source_set"]:
        digest = file_sha256(root, path)
        if digest is None:
            raise EvidenceError(f"{path} of the GPU sources cannot be read")
        digests[path] = digest
    return {
        "schema": EVIDENCE_SCHEMA,
        "decisions": register["decisions"],
        "recorded_at": recorded_at,
        "adapter": {"name": name, "api": api},
        "test": {"nodeid": GPU_TEST, "outcome": outcome},
        "parity": {
            "declared_input": DECLARED_INPUT,
            "result": "GPU packets equal CPU packets and decode back to the input, bit for bit",
        },
        "excepted_lines_covered": covered,
        "source_sha256": digests,
        "tool_versions": {
            "python": python_version,
            "sc_neurocore_engine": str(adapter.get("engine_version", "")),
            "coverage": str(coverage_report.get("meta", {}).get("version", "")),
        },
        "claim": "a physical-GPU run of the GPU encoder; no performance claim",
    }


def main(argv: list[str] | None = None) -> int:
    """Write the evidence record from one physical-GPU run's three reports.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; the process's own when None.

    Returns
    -------
    int
        0 when the record is written, 2 when an input refuses it.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True, type=Path, help="the adapter JSON")
    parser.add_argument("--junit", required=True, type=Path, help="the JUnit report")
    parser.add_argument(
        "--coverage-json", required=True, type=Path, help="the coverage json report"
    )
    parser.add_argument("--python-version", required=True, help="the interpreter that ran the test")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout")
    args = parser.parse_args(argv)
    try:
        adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
        coverage_report = json.loads(args.coverage_json.read_text(encoding="utf-8"))
        junit_text = args.junit.read_text(encoding="utf-8")
        register, _ = load_register(args.root)
        recorded_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = build_record(
            args.root, adapter, junit_text, coverage_report, args.python_version, recorded_at
        )
    except (OSError, ValueError) as error:
        print(f"gpu waveform encoder evidence: {error}", file=sys.stderr)
        return 2
    out = args.root / str(register["evidence"])
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"gpu waveform encoder evidence: recorded {record['adapter']['name']} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
