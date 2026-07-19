# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/41_one_model_five_views.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_one_model_five_views_has_honest_boundary_and_clean_schema() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])
    ids = [cell["id"] for cell in notebook["cells"]]

    assert len(ids) == len(set(ids))
    assert "does not prove" in text.lower()
    assert "HodgkinHuxleyNeuron" in text
    assert "MorrisLecarNeuron" in text
    assert "AdExNeuron" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])


def test_one_model_five_views_executes_catalogue_models() -> None:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    assert set(namespace["MODELS"]) == {
        "HodgkinHuxleyNeuron",
        "MorrisLecarNeuron",
        "AdExNeuron",
    }
    assert namespace["mse"] > 0.0
    assert namespace["errs"][-1] < 0.01
