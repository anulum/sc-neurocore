# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""TOML/JSON parity for the source MAT(1) schema."""

from __future__ import annotations
import json
import tomllib
from pathlib import Path


def test_source_schema_pair_is_identical() -> None:
    root = Path("src/sc_neurocore/neurons/model_schemas")
    toml = tomllib.loads((root / "non_resetting_lif.toml").read_text())
    data = json.loads((root / "non_resetting_lif.json").read_text())
    assert toml == data
    assert data["metadata"]["doi"] == "10.3389/neuro.10.009.2009"
    assert data["parameters"]["tau_theta"] == 50.0
    assert data["parameters"]["refractory_period"] == 2.0
