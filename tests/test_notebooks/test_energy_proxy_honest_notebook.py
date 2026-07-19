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
NOTEBOOK = REPO / "notebooks/46_energy_proxy_honest.ipynb"


def _load() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _source(cell: dict[str, Any]) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_energy_proxy_declares_toggle_only_boundary() -> None:
    notebook = _load()
    text = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "toggle counts" in text
    assert "not joules" in text
    assert 'backend="python"' in text
    assert all("id" in cell for cell in notebook["cells"])


def test_energy_proxy_counts_every_recorded_spike() -> None:
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), str(NOTEBOOK), "exec"), namespace)

    rows = namespace["rows"]
    regs_per_neuron = namespace["REGS_PER_NEURON"]
    assert all(event == spikes * regs_per_neuron for _, _, event, _, _, spikes in rows)
    assert all(clock > event > 0 for _, clock, event, _, _, _ in rows)
    assert all(rate > 0.0 for _, _, _, _, rate, _ in rows)
