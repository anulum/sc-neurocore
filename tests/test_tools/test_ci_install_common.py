# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared CI wheel admission

"""Refuse missing, ambiguous and ABI-incompatible shared engine artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools.ci_install_common import install_editable


@pytest.mark.parametrize("count", [0, 2])
def test_shared_engine_artifact_must_be_present_and_unambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """A missing or duplicate download cannot fall back to building another engine."""
    for index in range(count):
        (tmp_path / f"sc_neurocore_engine-3.16.{index}-cp312-cp312-linux_x86_64.whl").touch()
    monkeypatch.setenv("SC_NEUROCORE_CI_ENGINE_WHEEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=f"found {count}"):
        install_editable("dev")


def test_real_pip_refuses_an_incompatible_shared_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual pip ABI admission rejects the wrong platform before any package is installed."""
    (tmp_path / "sc_neurocore_engine-3.16.0-cp310-cp310-win_amd64.whl").touch()
    monkeypatch.setenv("SC_NEUROCORE_CI_ENGINE_WHEEL_DIR", str(tmp_path))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    with pytest.raises(subprocess.CalledProcessError):
        install_editable("dev")
