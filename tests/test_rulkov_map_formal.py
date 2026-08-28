# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov formal-lane contracts

"""Formal emission contracts for both Rulkov event identities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
EMITTER = ROOT / "tools" / "emit_catalogue_formal.py"
CATALOGUE = ROOT / "hdl" / "formal" / "catalogue"


def _load_emitter() -> ModuleType:
    name = "emit_catalogue_formal_rulkov_contract"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_both_event_identities_use_q1616_depth_four_jobs() -> None:
    """The 0.001 slow timescale must not quantise away in formal RTL."""
    emitter = _load_emitter()

    assert emitter.CLASS_TO_SCHEMA["RulkovMapNeuron"] == "rulkov_map"
    assert emitter.RETAINED_SC_CLASS_TO_SCHEMA["SCUpwardCrossingRulkovMapNeuron"] == (
        "sc_upward_crossing_rulkov_map"
    )
    for schema in ("rulkov_map", "sc_upward_crossing_rulkov_map"):
        assert emitter.PRECISION_BY_SCHEMA[schema] == (32, 16)
        assert emitter.DEPTH_BY_SCHEMA[schema] == 4
        assert schema in emitter.MINIMAL_SAFETY_SCHEMAS
    inventory = (CATALOGUE / "INVENTORY.md").read_text(encoding="utf-8")
    assert "| RulkovMapNeuron | rulkov_map | `sc_rulkov_map` | `x_out` | Q16.16 | 4 |" in inventory
    assert "| SCUpwardCrossingRulkovMapNeuron |" in inventory


def test_committed_jobs_bind_distinct_event_rtl() -> None:
    """Generated formal jobs must retain source-reset and SC-crossing events."""
    source = (CATALOGUE / "sc_rulkov_map.v").read_text(encoding="utf-8")
    retained = (CATALOGUE / "sc_upward_crossing_rulkov_map.v").read_text(encoding="utf-8")

    assert "Fixed-point: Q16.16" in source
    assert "Fixed-point: Q16.16" in retained
    assert "x_reg >= ((P_ALPHA + y_reg) + I_t)" in source
    assert "x_next >= P_X_THRESHOLD" in retained
    for module in ("sc_rulkov_map", "sc_upward_crossing_rulkov_map"):
        sby = (CATALOGUE / f"{module}.sby").read_text(encoding="utf-8")
        harness = (CATALOGUE / f"{module}_formal.v").read_text(encoding="utf-8")
        assert "depth 4" in sby
        assert "Minimal safety: async reset clears the spike flag" in harness
