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

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/48_fail_closed_gallery.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_fail_closed_gallery_names_real_production_surfaces() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "no mirrored validator" in text
    assert "SCNIRConversionConfig" in text
    assert "StochasticSTDPSynapse" in text
    assert "BitstreamEncoder" not in text
    assert all("id" in cell for cell in notebook["cells"])


def test_fail_closed_gallery_requires_four_real_refusals() -> None:
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    refusals = namespace["refusals"]
    assert set(refusals) == {
        "zero_bitstream_length",
        "fraction_not_below_width",
        "zero_stdp_window",
        "non_binary_pre_bit",
    }
    assert all(value.startswith("ValueError:") for value in refusals.values())
    assert namespace["valid_output"] in (0, 1)
