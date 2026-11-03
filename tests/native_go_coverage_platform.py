# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual unsupported-platform native-gate refusal

"""Use a real non-Linux runtime to check public refusals without running a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    """Retain genuine API and CLI refusals from the actual non-Linux CI runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--go", type=Path, required=True)
    parser.add_argument("--job-seconds", type=int, required=True)
    parser.add_argument("--python-coverage-config", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == "linux" or sys.flags.optimize or args.job_seconds < 1:
        raise RuntimeError("This contract requires a genuine unoptimized non-Linux runtime")
    repository = args.repository.resolve(strict=True)
    allocation = args.allocation.resolve(strict=True)
    compiler = args.go.resolve(strict=True)
    config = args.python_coverage_config.resolve(strict=True)
    sys.path.insert(0, str(repository))
    from tools.go_coverage_budget import derive_profile
    from tools.go_coverage_images import running_interpreter_identity
    from tools.go_coverage_trace import validate_config
    from tools.go_mat_coverage import execute

    config_sha = validate_config(config, repository)
    try:
        derive_profile(args.job_seconds, time.time(), allocation, 1 / 4, 1 / 200)
    except RuntimeError as error:
        if "actual Linux cgroup-v2" not in str(error):
            raise
    else:
        raise RuntimeError("The actual unsupported runtime received a Linux capacity profile")
    api_output, cli_output = allocation / "refused-native-api", allocation / "refused-native-cli"
    jobs = max(1, os.cpu_count() or 1)
    try:
        execute(
            repository, api_output, compiler, args.job_seconds, jobs, args.job_seconds / 200, config
        )
    except ValueError as error:
        if "Unoptimized Linux Python" not in str(error):
            raise
    else:
        raise RuntimeError("The actual unsupported API was allowed to begin native execution")
    command = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "coverage",
        "run",
        "--rcfile",
        str(config),
        str(repository / "tools/go_mat_coverage.py"),
        "--repository",
        str(repository),
        "--output",
        str(cli_output),
        "--go",
        str(compiler),
        "--seconds",
        str(args.job_seconds),
        "--jobs",
        str(jobs),
        "--stop-seconds",
        str(args.job_seconds / 200),
        "--python-coverage-config",
        str(config),
    ]
    actual = subprocess.run(command, capture_output=True, text=True, check=False)
    with (allocation / "actual-platform-refusals.json").open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "platform": sys.platform,
                    "python": running_interpreter_identity(),
                    "argv": command,
                    "returncode": actual.returncode,
                    "stdout": actual.stdout,
                    "stderr": actual.stderr,
                    "config_sha256": config_sha,
                    "api_output_exists": api_output.exists(),
                    "cli_output_exists": cli_output.exists(),
                    "native_backend_qualification": False,
                },
                indent=2,
            )
            + "\n"
        )
    if (
        actual.returncode != 1
        or "Unoptimized Linux Python" not in actual.stderr
        or api_output.exists()
        or cli_output.exists()
    ):
        raise RuntimeError("The actual unsupported CLI did not refuse before native allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
