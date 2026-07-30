# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import json
import tomllib
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["mckean", "sc_triangular_mckean"])
def test_mckean_toml_json_schema_parity(name: str) -> None:
    root = Path(__file__).parents[1] / "src/sc_neurocore/neurons/model_schemas"
    assert tomllib.loads((root / f"{name}.toml").read_text()) == json.loads(
        (root / f"{name}.json").read_text()
    )
