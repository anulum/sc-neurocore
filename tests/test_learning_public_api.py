# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learning package public API tests

"""Regression tests for the public learning package facade."""

from __future__ import annotations

import sc_neurocore.learning as learning


def test_learning_facade_exports_scheduler_and_callback_surfaces() -> None:
    """The learning package facade must expose documented schedulers and callbacks."""
    expected_symbols = {
        "CSVCallback",
        "CosineScheduler",
        "ExponentialScheduler",
        "StepScheduler",
        "TensorBoardCallback",
        "TrainingCallback",
        "WandBCallback",
        "WarmupCosineScheduler",
    }

    assert expected_symbols.issubset(set(learning.__all__))
    for name in expected_symbols:
        assert getattr(learning, name).__name__ == name
