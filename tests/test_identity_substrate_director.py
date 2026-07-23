# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDirector from former test_identity_substrate.py

"""Focused suite: TestDirector from former test_identity_substrate.py."""

from __future__ import annotations

from tests.identity_substrate_support import *  # noqa: F403

class TestDirector:
    def test_monitor_returns_metrics(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        metrics = director.monitor()
        assert "mean_rate" in metrics
        assert "cv" in metrics
        assert "fano" in metrics

    def test_diagnose_returns_list(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        problems = director.diagnose()
        assert isinstance(problems, list)

    def test_correct_does_not_crash(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        director.correct()

    def test_report_is_readable(self):
        sub = _make_substrate()
        stim = np.random.default_rng(0).uniform(5, 15, (100, N_CORTICAL))
        sub.run(duration=0.1, dt=0.001, stimuli_sequence=stim)
        director = DirectorController(sub)
        report = director.report()
        assert "Rate:" in report
        assert "CV:" in report
        assert "Diagnosis:" in report
