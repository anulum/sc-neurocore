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
import numpy as np

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/45_nir_passport.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_nir_passport_declares_toy_graph_boundary() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "toy LIF graph" in text
    assert "Does not prove" in text
    assert "nir package required" in text
    assert all("id" in cell for cell in notebook["cells"])


def test_nir_passport_executes_real_nir_round_trip() -> None:
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    assert namespace["spikes"].shape == (80, 5)
    assert namespace["size"] > 0
    assert np.array_equal(namespace["a"]["output"], namespace["b"]["output"])
