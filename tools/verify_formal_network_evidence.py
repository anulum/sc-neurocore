# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI formal network evidence verifier

"""Generate and validate formal network verification evidence for CI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.formal import FormalReportValidationError, validate_formal_network_report


SCHEMA_VERSION = "sc-neurocore.formal-network-evidence-check.v0.1"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class FormalEvidenceConfig:
    """Inputs for one generated formal network evidence check."""

    output: Path
    module_name: str = "dense_lif_frontier_fixture"
    input_width: int = 3
    output_width: int = 2
    state_width: int = 16
    output_index: int = 0
    window_cycles: int = 8
    max_spikes: int = 4
    refractory_cycles: int = 2
    formal_depth: int = 20
    formal_mode: str = "bmc"

    @property
    def report_path(self) -> Path:
        """Return the generated formal report path."""
        return self.output / "formal_rate_bound_report.json"


def build_formal_verify_command(
    config: FormalEvidenceConfig,
    *,
    sby_available: bool,
) -> list[str]:
    """Build the CLI command used for the evidence check."""
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.cli",
        "formal",
        "verify-network",
        "--module-name",
        config.module_name,
        "--input-width",
        str(config.input_width),
        "--output-width",
        str(config.output_width),
        "--state-width",
        str(config.state_width),
        "--output-index",
        str(config.output_index),
        "--window-cycles",
        str(config.window_cycles),
        "--max-spikes",
        str(config.max_spikes),
        "--refractory-cycles",
        str(config.refractory_cycles),
        "--formal-depth",
        str(config.formal_depth),
        "--formal-mode",
        config.formal_mode,
        "--output",
        str(config.output),
        "--out",
        str(config.report_path),
    ]
    if sby_available:
        command.append("--run-symbiyosys")
    return command


def run_formal_evidence_check(
    config: FormalEvidenceConfig,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run formal generation and validate the resulting report with artifact_root."""
    sby_available = shutil.which("sby") is not None
    command = build_formal_verify_command(config, sby_available=sby_available)
    completed = runner(command, capture_output=True, text=True, check=False)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "artifact_root": str(config.output),
        "report": str(config.report_path),
        "symbiyosys_requested": sby_available,
        "error": None,
    }
    if completed.returncode != 0:
        summary["error"] = f"formal verify-network exited with {completed.returncode}"
        return summary
    try:
        payload = json.loads(config.report_path.read_text(encoding="utf-8"))
        validate_formal_network_report(payload, artifact_root=config.output)
    except (OSError, json.JSONDecodeError, FormalReportValidationError, ValueError) as exc:
        summary["error"] = str(exc)
        return summary

    symbiyosys = payload.get("symbiyosys", {})
    summary["symbiyosys_status"] = (
        symbiyosys.get("status") if isinstance(symbiyosys, dict) else None
    )
    summary["passed"] = True
    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = subprocess.run,
) -> int:
    """Run the formal evidence verifier."""
    parser = argparse.ArgumentParser(
        description="Generate and validate SC-NeuroCore formal network evidence"
    )
    parser.add_argument("--output", type=Path, default=Path("build/formal-evidence"))
    parser.add_argument("--summary", type=Path, help="Write JSON check summary")
    parser.add_argument("--module-name", default="dense_lif_frontier_fixture")
    parser.add_argument("--input-width", type=int, default=3)
    parser.add_argument("--output-width", type=int, default=2)
    parser.add_argument("--state-width", type=int, default=16)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--window-cycles", type=int, default=8)
    parser.add_argument("--max-spikes", type=int, default=4)
    parser.add_argument("--refractory-cycles", type=int, default=2)
    parser.add_argument("--formal-depth", type=int, default=20)
    parser.add_argument("--formal-mode", choices=["bmc", "prove", "cover"], default="bmc")
    args = parser.parse_args(argv)

    config = FormalEvidenceConfig(
        output=args.output,
        module_name=args.module_name,
        input_width=args.input_width,
        output_width=args.output_width,
        state_width=args.state_width,
        output_index=args.output_index,
        window_cycles=args.window_cycles,
        max_spikes=args.max_spikes,
        refractory_cycles=args.refractory_cycles,
        formal_depth=args.formal_depth,
        formal_mode=args.formal_mode,
    )
    summary = run_formal_evidence_check(config, runner=runner)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
