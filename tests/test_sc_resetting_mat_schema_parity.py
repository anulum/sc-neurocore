# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - SC resetting-MAT paired-schema parity

from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import tomllib

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

SCHEMAS = Path("src/sc_neurocore/neurons/model_schemas")


def test_sc_resetting_mat_paired_schema_payloads_are_identical() -> None:
    with (SCHEMAS / "sc_resetting_mat.toml").open("rb") as handle:
        toml = tomllib.load(handle)
    with (SCHEMAS / "sc_resetting_mat.json").open(encoding="utf-8") as handle:
        json_schema = json.load(handle)
    assert toml == json_schema


def test_sc_resetting_mat_schema_matches_historical_trace() -> None:
    currents = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    hand = SCResettingMATNeuron()
    schema = UniversalNeuron.from_schema(SCHEMAS / "sc_resetting_mat.json")
    for current in currents:
        assert schema.step(I=current) == hand.step(current)
        np.testing.assert_allclose(
            [schema.state[key] for key in ("v", "theta1", "theta2")],
            [hand.v, hand.theta1, hand.theta2],
            rtol=0.0,
            atol=2.0e-12,
        )
