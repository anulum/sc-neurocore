# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""TOML/JSON parity for the retained project schema."""

from __future__ import annotations
import json

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path.
    import tomli as tomllib
from pathlib import Path


def test_project_schema_pair_is_identical() -> None:
    root = Path("src/sc_neurocore/neurons/model_schemas")
    toml = tomllib.loads((root / "sc_non_resetting_adaptive_lif.toml").read_text())
    data = json.loads((root / "sc_non_resetting_adaptive_lif.json").read_text())
    assert toml == data
    assert data["metadata"]["author"] == "SC-NeuroCore project"
    assert data["extensions"]["model_type"] == "sc_non_resetting_adaptive_lif"
