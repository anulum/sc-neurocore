# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Behaviour probe test support

"""Shared imports and synthetic observations for behavior-probe tests."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.behavior_taxonomy import BEHAVIOR_TAGS
from sc_neurocore.studio.behavior_probe import (
    BEHAVIOR_SWEEP_CURRENTS,
    BehaviorObservation,
    behavior_tags_for,
    derive_behavior_tags,
    probe_all_models,
    probe_model_behavior,
)

__all__ = [
    "BEHAVIOR_SWEEP_CURRENTS",
    "BEHAVIOR_TAGS",
    "_obs",
    "behavior_tags_for",
    "derive_behavior_tags",
    "probe_all_models",
    "probe_model_behavior",
    "pytest",
]


def _obs(
    current: float,
    pattern: str,
    *,
    rate_hz: float = 0.0,
    spike_count: int = 0,
    reproducible: bool = True,
    error: str | None = None,
) -> BehaviorObservation:
    """Construct a synthetic observation."""

    return BehaviorObservation(
        current=current,
        pattern=pattern,
        rate_hz=rate_hz,
        spike_count=spike_count,
        reproducible=reproducible,
        error=error,
    )
