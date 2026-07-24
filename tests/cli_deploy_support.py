# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cli_deploy.py

from __future__ import annotations


"""Exercise checkpoint and target deployment through the public CLI."""


import hashlib


import json


from pathlib import Path


import subprocess


import types


from unittest import mock


import pytest


from tests.cli_test_support import fake_module, run_cli


def _checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_checkpoint_deploy(
    checkpoint: Path,
    output: Path,
    *,
    digest: str | None,
    target: str = "ice40",
    bitstream_length: int = 64,
) -> int:
    arguments = [
        "deploy",
        str(checkpoint),
        "--target",
        target,
        "--T",
        str(bitstream_length),
        "--output",
        str(output),
    ]
    if digest is not None:
        arguments.extend(("--checkpoint-sha256", digest))
    return run_cli(*arguments)


def _synthesis_project(tmp_path: Path) -> Path:
    """Create the minimum HDL tree accepted by the synthesis adapter."""
    output = tmp_path / "synthesis"
    hdl = output / "hdl"
    hdl.mkdir(parents=True)
    (hdl / "fixture.v").write_text("module fixture; endmodule\n", encoding="utf-8")
    return output


__all__ = ['hashlib', 'json', 'Path', 'subprocess', 'types', 'mock', 'pytest', 'fake_module', 'run_cli', '_checkpoint_digest', '_run_checkpoint_deploy', '_synthesis_project']

