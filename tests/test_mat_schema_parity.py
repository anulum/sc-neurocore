# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - source MAT* paired-schema parity

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path.
    import tomli as tomllib

from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

SCHEMAS = Path("src/sc_neurocore/neurons/model_schemas")


def test_mat_paired_schema_payloads_are_identical() -> None:
    import json

    with (SCHEMAS / "mat.toml").open("rb") as handle:
        toml = tomllib.load(handle)
    with (SCHEMAS / "mat.json").open(encoding="utf-8") as handle:
        json_schema = json.load(handle)
    assert toml == json_schema


def test_mat_schema_matches_hand_model_complete_trace() -> None:
    currents = [0.0] * 32 + [0.5] * 5000 + [0.2, 0.7] * 512
    hand = MATNeuron()
    schema = UniversalNeuron.from_schema(SCHEMAS / "mat.toml")
    for current in currents:
        assert schema.step(I=current) == hand.step(current)
        np.testing.assert_allclose(
            [schema.state[key] for key in ("v", "theta1", "theta2", "refractory_remaining")],
            [hand.v, hand.theta1, hand.theta2, hand.refractory_remaining],
            rtol=0.0,
            atol=2.0e-12,
        )
