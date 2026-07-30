# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
import json
import tomllib
from pathlib import Path


def test_sc_sigma_delta_toml_json_schema_parity() -> None:
    root = Path(__file__).parents[1] / "src/sc_neurocore/neurons/model_schemas"
    toml = tomllib.loads((root / "sc_sigma_delta_accumulator.toml").read_text())
    data = json.loads((root / "sc_sigma_delta_accumulator.json").read_text())
    assert toml == data
