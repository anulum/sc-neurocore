# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training config

"""Focused suite: TestTrainingConfig from former test_studio_training.py."""

from __future__ import annotations

import time
from tests.studio_training_support import (
    get_training_status,
    list_surrogates,
    start_training,
)


class TestTrainingConfig:
    """Training runs with the default configuration."""

    def test_default_config_runs(self) -> None:
        """A default training run starts running and completes."""
        result = start_training({"epochs": 2, "batch_size": 32, "dataset": "synthetic"})
        assert result["status"] == "running"
        # Wait for completion (synthetic is fast)
        status = get_training_status(result["job_id"])
        for _ in range(20):
            time.sleep(1)
            status = get_training_status(result["job_id"])
            if status["status"] in ("completed", "failed"):
                break
        assert status["status"] == "completed", f"Expected completed, got {status}"

    def test_all_surrogates_listed(self) -> None:
        """Every surrogate is listed."""
        expected = {
            "fast_sigmoid",
            "superspike",
            "atan_surrogate",
            "sigmoid_surrogate",
            "straight_through",
            "triangular",
        }
        actual = {s["name"] for s in list_surrogates()}
        assert expected == actual
