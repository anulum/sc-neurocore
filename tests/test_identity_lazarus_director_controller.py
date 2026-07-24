# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDirectorController from former test_identity_lazarus.py

"""Focused suite: TestDirectorController from former test_identity_lazarus.py."""

from __future__ import annotations

from tests.identity_lazarus_support import *  # noqa: F403


class TestDirectorController:
    def test_monitor_returns_dict(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        metrics = director.monitor()
        assert isinstance(metrics, dict)

    def test_diagnose_returns_list(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        problems = director.diagnose()
        assert isinstance(problems, list)

    def test_report_returns_string(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        report = director.report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_correct_does_not_crash(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        director = DirectorController(sub)
        director.correct()  # should not raise
