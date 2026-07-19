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

from sc_neurocore.utils.fault_injection import FaultInjector

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/42_fault_tolerance_theatre.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_fault_tolerance_theatre_refuses_fixed_point_overclaim() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "production `FaultInjector.inject_bit_flips`" in text
    assert "Does not prove" in text
    assert "Superiority over fixed-point arithmetic" in text
    assert "float + relative noise" not in text
    assert all("id" in cell for cell in notebook["cells"])


def test_fault_tolerance_theatre_calls_real_fault_injector(monkeypatch: Any) -> None:
    calls = 0
    original = FaultInjector.inject_bit_flips

    def tracked(bitstream: np.ndarray[Any, Any], error_rate: float) -> np.ndarray[Any, Any]:
        nonlocal calls
        calls += 1
        return original(bitstream, error_rate)

    monkeypatch.setattr(FaultInjector, "inject_bit_flips", tracked)
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    assert calls == len(namespace["bers"]) * namespace["trials"] * 2
    assert namespace["sc_errs"][-1] > namespace["sc_errs"][0]
    assert namespace["spikes"] > 0
