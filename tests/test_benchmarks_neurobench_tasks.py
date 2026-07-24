# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTasks from former test_benchmarks_neurobench.py

"""Focused suite: TestTasks from former test_benchmarks_neurobench.py."""

from __future__ import annotations

from tests.benchmarks_neurobench_support import *  # noqa: F403


class TestTasks:
    def test_task_registry(self):
        assert len(TASKS) >= 5
        assert "mnist" in TASKS
        assert "dvs_gesture" in TASKS
        assert "keyword_spotting" in TASKS

    def test_task_fields(self):
        t = TASKS["mnist"]
        assert isinstance(t, BenchmarkTask)
        assert t.n_classes == 10
        assert t.metric == "accuracy"
        assert t.input_shape == (784,)
        assert t.baseline_accuracy > 0

    def test_all_tasks_frozen(self):
        for name, task in TASKS.items():
            with pytest.raises(AttributeError):
                task.name = "mutated"

    def test_shd_task(self):
        t = TASKS["shd"]
        assert t.n_classes == 20
        assert t.neurobench_id == "shd"

    def test_heartbeat_task(self):
        t = TASKS["heartbeat_anomaly"]
        assert t.n_classes == 2
