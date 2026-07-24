# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunnerErrors from former test_equivalence_check.py

"""Focused suite: TestRunnerErrors from former test_equivalence_check.py."""

from __future__ import annotations

from tests.equivalence_check_support import *  # noqa: F403


class TestRunnerErrors:
    """Error handling that does not require a real solver run."""

    def test_timeout_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import subprocess

        monkeypatch.setattr(_sby_runner.shutil, "which", lambda name: "/usr/bin/x")

        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(_sby_runner.subprocess, "run", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out"):
            prove_equivalence(
                _TINY_DUT,
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
                timeout_s=1.0,
                workdir=tmp_path,
            )
