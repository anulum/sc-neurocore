# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine ABI capture format and drift contracts

"""Reject damaged real capture files and detect changes to persisted class contracts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.engine_abi_contracts import JsonValue, differences, read_inventory, validate_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def captured_contract_inventory(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Capture the actual interface once for independent damaged-file tests.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-owned temporary-directory factory.

    Returns
    -------
    bytes
        Immutable bytes returned by the actual public CLI.
    """
    path = tmp_path_factory.mktemp("engine-abi-contracts") / "actual-engine.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/engine_abi_inventory.py"),
            "capture",
            "--output",
            str(path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return path.read_bytes()


@pytest.fixture
def live_capture(tmp_path: Path, captured_contract_inventory: bytes) -> Path:
    """Create a test-owned copy before exercising a persisted contract change.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned temporary directory.
    captured_contract_inventory : bytes
        Unmodified interface bytes captured from the real engine.

    Returns
    -------
    pathlib.Path
        Independent persisted capture.
    """
    path = tmp_path / "actual-engine.json"
    path.write_bytes(captured_contract_inventory)
    return path


def test_compare_detects_actual_class_member_removal(live_capture: Path) -> None:
    """Retain the full live inventory and remove only its actual step contract."""
    before = read_inventory(live_capture)
    after = copy.deepcopy(before)
    modules = after["modules"]
    assert isinstance(modules, dict)
    native = modules["sc_neurocore_engine.sc_neurocore_engine"]
    assert isinstance(native, dict)
    neuron = native["FixedPointLif"]
    assert isinstance(neuron, dict)
    members = neuron["members"]
    assert isinstance(members, dict)
    del members["step"]
    assert differences(before, after) == [
        "inventory.modules.sc_neurocore_engine.sc_neurocore_engine.FixedPointLif.members.step: removed"
    ]


def test_compare_detects_actual_pickle_identity_change(live_capture: Path) -> None:
    """Report a change to a live class's persisted module identity."""
    before = read_inventory(live_capture)
    after = copy.deepcopy(before)
    modules = after["modules"]
    assert isinstance(modules, dict)
    native = modules["sc_neurocore_engine.sc_neurocore_engine"]
    assert isinstance(native, dict)
    neuron = native["FixedPointLif"]
    assert isinstance(neuron, dict)
    neuron["__module__"] = "sc_neurocore_engine"
    changes = differences(before, after)
    assert len(changes) == 1
    assert "FixedPointLif.__module__: changed" in changes[0]


@pytest.mark.parametrize("schema", [True, 0, 2, "1", None])
def test_read_refuses_invalid_schema(live_capture: Path, schema: JsonValue) -> None:
    """Refuse malformed versions in an otherwise complete real capture."""
    raw = json.loads(live_capture.read_text())
    raw["schema"] = schema
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires schema 1"):
        read_inventory(live_capture)


def test_read_refuses_empty_namespace_inventory(live_capture: Path) -> None:
    """Reject a damaged capture whose namespaces have been erased."""
    raw = json.loads(live_capture.read_text())
    raw["inventory"]["modules"] = {}
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires both namespaces"):
        read_inventory(live_capture)


def test_read_refuses_absent_advertised_export(live_capture: Path) -> None:
    """Reject a real facade capture after one advertised object is removed."""
    raw = json.loads(live_capture.read_text())
    del raw["inventory"]["modules"]["sc_neurocore_engine"]["FixedPointLif"]
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="names an absent symbol"):
        read_inventory(live_capture)


def test_read_refuses_nonfinite_capture_value(live_capture: Path) -> None:
    """Reject nonstandard NaN data in a real capture's provenance."""
    raw = json.loads(live_capture.read_text())
    raw["provenance"]["platform"] = float("nan")
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="finite numbers"):
        read_inventory(live_capture)


@pytest.mark.parametrize("side", ["before", "after", "both"])
@pytest.mark.parametrize("conflicting", [False, True])
@pytest.mark.parametrize(
    "section,key",
    [
        ((), "schema"),
        (("provenance",), "python"),
        (("inventory",), "modules"),
        (("inventory", "modules"), "sc_neurocore_engine.sc_neurocore_engine"),
        (
            ("inventory", "modules", "sc_neurocore_engine.sc_neurocore_engine", "FixedPointLif"),
            "__module__",
        ),
        (
            (
                "inventory",
                "modules",
                "sc_neurocore_engine.sc_neurocore_engine",
                "FixedPointLif",
                "members",
            ),
            "step",
        ),
        (
            (
                "inventory",
                "modules",
                "sc_neurocore_engine.sc_neurocore_engine",
                "FixedPointLif",
                "global_reference",
            ),
            "same_object",
        ),
    ],
)
def test_cli_refuses_duplicate_capture_keys(
    live_capture: Path, section: tuple[str, ...], key: str, conflicting: bool, side: str
) -> None:
    """Refuse ambiguous observations at every capture layer on either CLI input."""
    raw = json.loads(live_capture.read_text())
    target = raw
    for name in section:
        target = target[name]
    first_value = None if conflicting else target[key]
    recorded_fields = list(target.items())
    target.clear()
    target["\0" + key] = first_value
    target.update(recorded_fields)
    encoded = json.dumps(raw)
    marker = json.dumps("\0" + key) + ": "
    assert encoded.count(marker) == 1
    damaged = live_capture.with_name("duplicate-fields.json")
    damaged.write_text(encoded.replace(marker, json.dumps(key) + ": ", 1))
    with pytest.raises(ValueError, match="duplicate object keys"):
        read_inventory(damaged)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/engine_abi_inventory.py"),
            "compare",
            "--before",
            str(live_capture if side == "after" else damaged),
            "--after",
            str(live_capture if side == "before" else damaged),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "duplicate object keys" in result.stderr
    assert "inventories match" not in result.stdout


def test_validator_refuses_filesystem_object(live_capture: Path) -> None:
    """Reject an actual filesystem path passed to the public JSON validator."""
    with pytest.raises(ValueError, match="JSON values"):
        validate_json(live_capture)


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("inventory", None, "no inventory object"),
        ("modules", None, "requires modules and facade_exports"),
        ("facade_exports", None, "requires modules and facade_exports"),
        ("facade_exports", [], "nonempty named exports"),
        ("facade_exports", [42], "nonempty named exports"),
    ],
)
def test_read_refuses_damaged_inventory_fields(
    live_capture: Path, field: str, value: JsonValue, diagnostic: str
) -> None:
    """Refuse a real capture whose required namespace/export payload is damaged."""
    raw = json.loads(live_capture.read_text())
    target = raw if field == "inventory" else raw["inventory"]
    target[field] = value
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=diagnostic):
        read_inventory(live_capture)


def test_read_refuses_namespace_without_version(live_capture: Path) -> None:
    """Reject an otherwise complete namespace after its version record is lost."""
    raw = json.loads(live_capture.read_text())
    del raw["inventory"]["modules"]["sc_neurocore_engine"]["__version__"]
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no versioned symbol inventory"):
        read_inventory(live_capture)


def test_read_refuses_symbol_without_kind(live_capture: Path) -> None:
    """Reject an incomplete actual native class record before claiming equality."""
    raw = json.loads(live_capture.read_text())
    del raw["inventory"]["modules"]["sc_neurocore_engine"]["FixedPointLif"]["kind"]
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="has no kind"):
        read_inventory(live_capture)


def test_read_accepts_finite_provenance_measurement(live_capture: Path) -> None:
    """Accept an actual filesystem timestamp without changing the interface."""
    before = read_inventory(live_capture)
    raw = json.loads(live_capture.read_text())
    raw["provenance"]["capture_mtime"] = live_capture.stat().st_mtime
    live_capture.write_text(json.dumps(raw))
    assert read_inventory(live_capture) == before


def test_validator_refuses_nonstring_inventory_key(live_capture: Path) -> None:
    """Refuse a filesystem object used as a key in a real namespace payload."""
    raw = json.loads(live_capture.read_text())
    inventory = raw["inventory"]
    inventory[live_capture] = inventory.pop("modules")
    with pytest.raises(ValueError, match="string keys"):
        validate_json(inventory)


@pytest.mark.parametrize("change", ["type", "alias", "order"])
def test_compare_detects_export_metadata_drift(live_capture: Path, change: str) -> None:
    """Detect scalar type, new alias and ordered export changes in a real capture."""
    before = read_inventory(live_capture)
    raw = json.loads(live_capture.read_text())
    inventory = raw["inventory"]
    native = inventory["modules"]["sc_neurocore_engine.sc_neurocore_engine"]
    if change == "type":
        native["__version__"]["constant"] = 1
        expected = ".__version__.constant: changed"
    elif change == "alias":
        native["popcount_alias"] = copy.deepcopy(native["popcount"])
        expected = ".popcount_alias: added"
    else:
        inventory["facade_exports"].reverse()
        expected = ".facade_exports: changed"
    live_capture.write_text(json.dumps(raw))
    changes = differences(before, read_inventory(live_capture))
    assert len(changes) == 1
    assert expected in changes[0]


@pytest.mark.parametrize(
    "field",
    [
        "__module__",
        "__qualname__",
        "__name__",
        "__text_signature__",
        "signature",
        "signature_error",
        "members",
        "global_reference",
        "aliases",
    ],
)
def test_cli_refuses_incomplete_identical_class_captures(live_capture: Path, field: str) -> None:
    """Refuse equal damaged captures instead of claiming absent ABI observations."""
    raw = json.loads(live_capture.read_text())
    del raw["inventory"]["modules"]["sc_neurocore_engine.sc_neurocore_engine"]["FixedPointLif"][
        field
    ]
    live_capture.write_text(json.dumps(raw))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/engine_abi_inventory.py"),
            "compare",
            "--before",
            str(live_capture),
            "--after",
            str(live_capture),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Engine ABI inventory failed" in result.stderr
    assert "inventories match" not in result.stdout


@pytest.mark.parametrize(
    "fault",
    [
        "identity-type",
        "signature-type",
        "error-type",
        "signature-both-null",
        "signature-both-observed",
        "members-type",
        "member-kind",
        "member-identity",
        "member-signature",
        "reference-type",
        "reference-fields",
        "reference-boolean",
        "reference-error-type",
        "successful-reference-error",
        "aliases-type",
        "aliases-empty",
        "aliases-element",
        "aliases-empty-name",
        "aliases-duplicate",
        "duplicate-export",
    ],
)
def test_read_refuses_damaged_live_symbol_metadata(live_capture: Path, fault: str) -> None:
    """Keep each actual class and method observation complete and well typed."""
    raw = json.loads(live_capture.read_text())
    record = raw["inventory"]["modules"]["sc_neurocore_engine.sc_neurocore_engine"]["FixedPointLif"]
    changes = {
        "identity-type": ("__module__", 1),
        "signature-type": ("signature", 1),
        "error-type": ("signature_error", 1),
        "signature-both-null": ("signature", None),
        "signature-both-observed": ("signature_error", "observed failure"),
        "members-type": ("members", []),
        "reference-type": ("global_reference", []),
        "reference-fields": ("global_reference", {"same_object": True}),
        "reference-boolean": ("global_reference", {"same_object": "true", "error": None}),
        "reference-error-type": ("global_reference", {"same_object": False, "error": 1}),
        "successful-reference-error": (
            "global_reference",
            {"same_object": True, "error": "observed failure"},
        ),
        "aliases-type": ("aliases", None),
        "aliases-empty": ("aliases", []),
        "aliases-element": ("aliases", [1]),
        "aliases-empty-name": ("aliases", [""]),
        "aliases-duplicate": ("aliases", ["sc_neurocore_engine.FixedPointLif"] * 2),
    }
    if fault in changes:
        field, value = changes[fault]
        record[field] = value
    elif fault.startswith("member-"):
        field = {
            "member-kind": "kind",
            "member-identity": "__module__",
            "member-signature": "signature",
        }[fault]
        del record["members"]["step"][field]
    else:
        raw["inventory"]["facade_exports"].append(raw["inventory"]["facade_exports"][0])
    live_capture.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="engine ABI"):
        read_inventory(live_capture)
