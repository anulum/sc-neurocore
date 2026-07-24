# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_observation_loader.py

from __future__ import annotations

"""Support extracted from test_observation_loader.py."""

import json


import pytest


from sc_neurocore.optimizer import (
    load_observations,
    load_synthesis_observation,
    observation_from_synthesis_reports,
    observations_from_payload,
)


from sc_neurocore.optimizer.observation_loader import ObservationLoadError


from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile


from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateSCOptimizer,
    TargetHardwareProfile,
)


def _design() -> dict[str, object]:
    return {
        "mac_count": 256,
        "bitstream_length": 128,
        "decorrelator": "LFSR",
        "mode": "SC",
        "precision_bits": 8,
        "lfsr_polynomial": "x16+x15+x13+x4+1",
        "is_critical_path": True,
    }



__all__ = ['json', 'pytest', 'load_observations', 'load_synthesis_observation', 'observation_from_synthesis_reports', 'observations_from_payload', 'ObservationLoadError', 'HardwareBudget', 'LayerProfile', 'SurrogateSCOptimizer', 'TargetHardwareProfile', '_design']
