# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit complete native gate contract consumer

"""Run each dedicated module contract against the same current native artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    """Refuse any failed or skipped dedicated native-tool contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Native contract assertions must remain enabled")
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    repository = args.repository.resolve(strict=True)
    job = args.job.resolve(strict=True)
    sys.path[:0] = [str(repository), str(repository / "src")]
    import pytest

    directory = repository / "tests/native_go_coverage_contracts"
    names = (
        "build",
        "build_inputs",
        "images",
        "image_inputs",
        "environment",
        "terminal_environment",
        "records",
        "record_inputs",
        "controls",
        "consumer",
        "gate",
        "gate_inputs",
        "packaging",
        "budget",
        "trace",
    )
    status = pytest.main(
        [
            "-q",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(directory),
            "--confcutdir",
            str(directory),
            "--basetemp",
            str(job / "contracts-tmp"),
            "--junitxml",
            str(job / "contracts.xml"),
            "--native-go-job",
            str(job),
            "--native-go-repository",
            str(repository),
            *(str(directory / (name + "_contract.py")) for name in names),
        ]
    )
    cases = ET.parse(job / "contracts.xml").findall(".//testcase")
    if (
        status != 0
        or not cases
        or any(
            case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
        )
    ):
        raise RuntimeError("A dedicated actual native-tool contract failed or skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
