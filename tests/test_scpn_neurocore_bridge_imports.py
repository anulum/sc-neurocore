# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN bridge import-surface contracts

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


from scpn_neurocore.bridge import (
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
    validate_qpu_artifact_payload,
)


def test_expected_import_surface() -> None:
    assert callable(load_connectome)
    assert callable(load_tokamak_data)
    assert callable(load_power_grid)
    assert callable(load_live_stream)
    assert callable(validate_qpu_artifact_payload)


def test_bridge_import_does_not_eagerly_require_datastream_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scpn_neurocore.bridge import load_live_stream; print(load_live_stream.__name__)",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "load_live_stream"
