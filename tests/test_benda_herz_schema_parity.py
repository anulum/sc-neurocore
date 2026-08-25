# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).parents[1] / "src/sc_neurocore/neurons/model_schemas"


def test_source_schema_json_toml_parity() -> None:
    assert json.loads((ROOT / "benda_herz.json").read_text()) == tomllib.loads(
        (ROOT / "benda_herz.toml").read_text()
    )


def test_sc_schema_json_toml_parity() -> None:
    assert json.loads((ROOT / "sc_stochastic_rate_adaptation.json").read_text()) == tomllib.loads(
        (ROOT / "sc_stochastic_rate_adaptation.toml").read_text()
    )
