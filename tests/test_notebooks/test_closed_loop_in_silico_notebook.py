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
NOTEBOOK = REPO / "notebooks/47_closed_loop_in_silico.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_closed_loop_uses_production_codec_without_fallback() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "codec.compress(source_raster)" in text
    assert "codec.decompress" in text
    assert "except Exception" not in text
    assert "no fallback" in text
    assert all("id" in cell for cell in notebook["cells"])


def test_closed_loop_executes_bit_exact_codec_path() -> None:
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    assert namespace["codec_round_trip_exact"] is True
    assert np.array_equal(namespace["source_raster"], namespace["restored_raster"])
    assert namespace["codec_stats"].n_spikes == int(namespace["source_raster"].sum())
    assert namespace["tracking_mae"] < 0.25
