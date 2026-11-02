# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Prepared assertion-enable graph contracts

"""Exercise the graph transform on an actual prepared catalogue proof."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy

import pytest

from tests.formal_native_evidence_support import collect_native
from tools.formal_assertion_enables import enable_model


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Serialize the actual native preparation of a tracked catalogue job."""
    root = tmp_path_factory.mktemp("enable-transform-native")
    collect_native(root)
    target = root / "prepared.json"
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_rtlil {root}/proof/model/design_prep.il; flatten; write_json {target}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return target.read_text()


def test_actual_graph_changes_only_assertions_and_existing_covers(prepared: str) -> None:
    """Keep every assumption, enable, initializer and non-property cell byte-for-byte."""
    result = enable_model(prepared)
    original = json.loads(prepared)
    converted = json.loads(result.serialized)
    cells = original["modules"][result.top]["cells"]
    output = converted["modules"][result.top]["cells"]
    assert result.goals
    for goal, identity in result.goals.items():
        name = identity.removeprefix(result.top + "/")
        expected = deepcopy(cells[name])
        expected["type"] = "$cover"
        expected["connections"]["A"] = ["1"]
        expected["attributes"]["hdlname"] = goal
        assert output.pop(goal) == expected
        cells.pop(name)
    for name in list(cells):
        if cells[name]["type"] == "$cover":
            del cells[name]
    assert converted == original
    assert enable_model(prepared).serialized == result.serialized


def test_preexisting_cover_cannot_substitute_for_an_assertion_enable(prepared: str) -> None:
    """Remove an unrelated native cover while retaining every assertion's own goal."""
    design = json.loads(prepared)
    model = enable_model(prepared)
    cells = design["modules"][model.top]["cells"]
    unrelated = deepcopy(next(cell for cell in cells.values() if cell["type"] == "$assert"))
    unrelated["type"] = "$cover"
    cells["unrelated_cover"] = unrelated
    result = enable_model(json.dumps(design))
    assert result.goals == model.goals
    assert "unrelated_cover" not in json.loads(result.serialized)["modules"][model.top]["cells"]


@pytest.mark.parametrize("kind", ["$check", "$live", "$fair", "$print"])
def test_mixed_native_model_cannot_hide_unhandled_formal_cells(prepared: str, kind: str) -> None:
    """Keep a reachable assertion and refuse an additional unsupported formal cell."""
    design = json.loads(prepared)
    top = enable_model(prepared).top
    cells = design["modules"][top]["cells"]
    extra = deepcopy(next(cell for cell in cells.values() if cell["type"] == "$assert"))
    extra["type"] = kind
    if kind == "$print":
        extra["parameters"]["FLAVOR"] = "assert"
    cells["unsupported_formal"] = extra
    with pytest.raises(ValueError, match="unsupported formal"):
        enable_model(json.dumps(design))


@pytest.mark.parametrize("defect", ["no-top", "two-tops", "hierarchy", "zero", "wide", "collision"])
def test_malformed_or_empty_native_models_refuse(prepared: str, defect: str) -> None:
    """Damage only the named native structure and require a closed refusal."""
    native = json.loads(prepared)
    top = enable_model(prepared).top
    module = native["modules"][top]
    cells = module["cells"]
    if defect == "no-top":
        del module["attributes"]["top"]
    elif defect == "two-tops":
        native["modules"]["duplicate_top"] = deepcopy(module)
    elif defect == "hierarchy":
        next(iter(cells.values()))["type"] = "unflattened_child"
    elif defect == "zero":
        module["cells"] = {name: cell for name, cell in cells.items() if cell["type"] != "$assert"}
    elif defect == "wide":
        next(cell for cell in cells.values() if cell["type"] == "$assert")["connections"]["EN"] = [
            "0",
            "1",
        ]
    else:
        cells["scn_assertion_enable_000000"] = deepcopy(next(iter(cells.values())))
    with pytest.raises(ValueError):
        enable_model(json.dumps(native))
