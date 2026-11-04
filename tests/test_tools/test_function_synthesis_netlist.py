# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native function netlist reader regressions

"""Validate complete native netlist records and their original input boundaries."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_tools.test_function_synthesis_words import WordCase
from tests.test_tools.test_function_synthesis_words import word_case as word_case
from tools.function_synthesis_netlist import Netlist, parameter_integer


def test_netlist_clone_does_not_mutate_the_actual_acquisition(word_case: WordCase) -> None:
    """Keep the retained graph isolated when a consumer edits its public export clone."""
    _, original, _ = word_case
    before = copy.deepcopy(original.raw)
    clone = original.clone_module()
    clone.clear()
    assert original.raw == before


@pytest.mark.parametrize("mutation", ["connection", "offset"])
def test_port_and_named_wire_views_must_describe_the_same_actual_hardware(
    word_case: WordCase, tmp_path: Path, mutation: str
) -> None:
    """Refuse contradictory port aliases before a native JSON export can reinterpret them."""
    document = json.loads((word_case[0] / "pre.json").read_text())
    result = document["modules"]["function_unit"]["netnames"]["result"]
    if mutation == "connection":
        result["bits"][0] = 999999
    else:
        result["offset"] = 1
    path = tmp_path / "contradictory-port.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="named original port"):
        Netlist.read(path)


def test_original_port_domain_does_not_depend_on_an_optional_named_alias(
    word_case: WordCase, tmp_path: Path
) -> None:
    """Keep all primary bits when JSON describes them directly without redundant wire names."""
    document = json.loads((word_case[0] / "pre.json").read_text())
    del document["modules"]["function_unit"]["netnames"]["argument0"]
    path = tmp_path / "direct-primary-port.json"
    path.write_text(json.dumps(document))
    assert Netlist.read(path).primary_inputs({"argument0": 8}, 8) == word_case[1].primary_inputs(
        {"argument0": 8}, 8
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "list",
        "direction",
        "boolean",
        "constant",
        "offset",
        "parameter",
        "connections",
        "empty-word",
        "empty-type",
    ],
)
def test_malformed_actual_netlist_is_refused(
    word_case: WordCase, tmp_path: Path, mutation: str
) -> None:
    """Reject damaged typed fields at the public reader before they can enter a proof."""
    root, _, _ = word_case
    document = json.loads((root / "pre.json").read_text())
    module = document["modules"]["function_unit"]
    cell = next(iter(module["cells"].values()))
    if mutation == "missing":
        del document["modules"]["function_unit"]
    elif mutation == "list":
        document["modules"] = []
    elif mutation == "direction":
        module["ports"]["argument0"]["direction"] = "inout"
    elif mutation == "boolean":
        module["ports"]["argument0"]["bits"][0] = True
    elif mutation == "constant":
        module["ports"]["argument0"]["bits"][0] = "2"
    elif mutation == "offset":
        next(iter(module["netnames"].values()))["offset"] = True
    elif mutation == "parameter":
        cell["parameters"]["invalid_parameter"] = 1
    elif mutation == "empty-word":
        module["ports"]["argument0"]["bits"] = []
    elif mutation == "empty-type":
        cell["type"] = ""
    else:
        cell["connections"].clear()
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        Netlist.read(path)


def test_linked_netlist_is_not_an_actual_acquisition(word_case: WordCase, tmp_path: Path) -> None:
    """Refuse a linked checkpoint even when its target is genuine native output."""
    linked = tmp_path / "linked.json"
    linked.symlink_to(word_case[0] / "pre.json")
    with pytest.raises(ValueError, match="regular"):
        Netlist.read(linked)


@pytest.mark.parametrize("value", ["", "x", "2", "-1"])
def test_parameters_require_exact_binary_integer_values(value: str) -> None:
    """Refuse nonbinary values instead of coercing a native parameter's semantics."""
    with pytest.raises(ValueError, match="binary integer"):
        parameter_integer({"WIDTH": value}, "WIDTH")


def test_missing_binary_parameter_is_not_zero() -> None:
    """Keep an absent required parameter distinct from an actual zero-valued one."""
    with pytest.raises(ValueError, match="binary integer"):
        parameter_integer({}, "WIDTH")
    assert parameter_integer({"WIDTH": "00101"}, "WIDTH") == 5


@pytest.mark.parametrize(
    "widths,output",
    [({}, 8), ({"argument0": True}, 8), ({"argument0": 8}, 65), ({"another": 8}, 8)],
)
def test_original_interface_requires_every_complete_independent_argument(
    word_case: WordCase, widths: dict[str, int], output: int
) -> None:
    """Refuse omitted, unsupported and substituted original primary domains."""
    with pytest.raises(ValueError):
        word_case[1].primary_inputs(widths, output)


@pytest.mark.parametrize("mutation", ["alias", "constant"])
def test_actual_primary_bits_cannot_be_aliased_or_constrained(
    word_case: WordCase, tmp_path: Path, mutation: str
) -> None:
    """Keep every original primary bit universally independent in a retained graph."""
    document = json.loads((word_case[0] / "pre.json").read_text())
    bits = document["modules"]["function_unit"]["ports"]["argument0"]["bits"]
    bits[0] = bits[1] if mutation == "alias" else "0"
    document["modules"]["function_unit"]["netnames"]["argument0"]["bits"] = list(bits)
    path = tmp_path / "constrained.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="constrained or aliased"):
        Netlist.read(path).primary_inputs({"argument0": 8}, 8)


@pytest.mark.parametrize("damage", ["duplicate-key", "NaN", "Infinity", "overflow", "nesting"])
def test_ambiguous_native_json_cannot_reinterpret_an_actual_graph(
    word_case: WordCase, tmp_path: Path, damage: str
) -> None:
    """Reject duplicate and nonfinite native values at the public graph admission boundary."""
    source = (word_case[0] / "pre.json").read_bytes()
    if damage == "duplicate-key":
        payload = b'{"modules":{},' + source.lstrip()[1:]
    elif damage == "nesting":
        payload = b"[" * 2000 + b"0" + b"]" * 2000
    else:
        token = "1e999" if damage == "overflow" else damage
        payload = b'{"unbound":' + token.encode() + b"," + source.lstrip()[1:]
    path = tmp_path / "refused.json"
    path.write_bytes(payload)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError):
        Netlist.read(path)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_native_file_growing_after_stat_is_still_bounded(
    word_case: WordCase, tmp_path: Path
) -> None:
    """Bound actual bytes read when a real acquisition grows between metadata and open."""
    path = tmp_path / "growing.json"
    shutil.copyfile(word_case[0] / "pre.json", path)
    script = """from pathlib import Path
import sys
from tools.function_synthesis_json import MAX_NETLIST_BYTES
from tools.function_synthesis_netlist import Netlist
path = Path(sys.argv[1])
armed = True
def observe(event, args):
    global armed
    if armed and event == "open" and args[0] == str(path):
        armed = False
        with path.open("ab") as output:
            output.truncate(MAX_NETLIST_BYTES + 1)
sys.addaudithook(observe)
try:
    Netlist.read(path)
except ValueError as error:
    assert "bounded regular artifact" in str(error)
else:
    raise AssertionError("growing native JSON escaped its parser byte budget")
assert not armed
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=20
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
