# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Refuse damaged retained execution artifacts

"""Use real recorder output to check public projection and custody refusals."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools.facet_receipt import main
from tools.facet_receipt_evidence import retain_execution

_VALIDATOR = (
    "tests/test_reference_mcculloch_pitts.py"
    "::test_truth_rows_match_an_independent_absolute_veto_rule"
)


@pytest.fixture(scope="module")
def actual_validator_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the real command-line recorder and return its original artifacts."""
    import sys

    root = tmp_path_factory.mktemp("real-validator-provenance")
    assert (
        main(
            [
                "record",
                "--model",
                "McCullochPittsNeuron",
                "--facet",
                "class_validated",
                "--receipt-dir",
                str(root / "receipts"),
                "--raw-evidence-dir",
                str(root / "private"),
                "--",
                sys.executable,
                "-m",
                "pytest",
                _VALIDATOR,
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        )
        == 0
    )
    archives = list((root / "private").glob("validator-*"))
    assert len(archives) == 1
    return archives[0]


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("xml", "invalid private JUnit report"),
        ("doctype", "invalid private JUnit report"),
        ("missing-runtime", "complete runtime provenance"),
        ("duplicate-runtime", "duplicate validator runtime property"),
        ("executable-digest", "invalid validator executable digest"),
        ("public-runtime", "invalid public validator runtime fact"),
        ("testcase-identity", "testcase identifier"),
        ("result-accounting", "JUnit result accounting"),
        ("unclassified-location", "unclassified private location"),
        ("root-file", "unclassified private location"),
        ("windows-location", "unclassified private location"),
        ("network-share", "unclassified private location"),
    ],
)
def test_damaged_real_execution_keeps_raw_custody_without_public_credit(
    tmp_path: Path, actual_validator_archive: Path, damage: str, reason: str
) -> None:
    """Reject altered real artifacts without losing the exact rejected bytes."""
    from tools.facet_receipt import REPO_ROOT

    junit = (actual_validator_archive / "junit.xml").read_bytes()
    stdout = (actual_validator_archive / "stdout.bin").read_bytes()
    root = ElementTree.fromstring(junit)
    properties = root.find(".//properties")
    assert properties is not None
    properties.insert(
        0, ElementTree.Element("property", {"name": "archive.note", "value": "refusal-control"})
    )
    ElementTree.indent(root)
    locations = {
        "unclassified-location": b"/unclassified/private/location",
        "root-file": b"/private-recording",
        "windows-location": b"C:\\private\\recording",
        "network-share": b"\\\\private-host\\recording",
    }
    if damage == "xml":
        junit = b"<testsuites"
    elif damage == "doctype":
        junit = b"<!DOCTYPE testsuites>" + ElementTree.tostring(root)
    elif damage in locations:
        stdout += b"\ndiagnostic " + locations[damage] + b"\n"
    else:
        for item in list(properties):
            if item.get("name") == "facet_runtime.python" and damage == "missing-runtime":
                properties.remove(item)
            elif item.get("name") == "facet_runtime.python" and damage == "duplicate-runtime":
                properties.append(copy.deepcopy(item))
            elif (
                item.get("name") == "facet_runtime.executable_sha256"
                and damage == "executable-digest"
            ):
                item.set("value", "invalid")
            elif item.get("name") == "facet_runtime.python" and damage == "public-runtime":
                item.set("value", "/home/private/identity")
        if damage in {"testcase-identity", "result-accounting"}:
            case = root.find(".//testcase")
            assert case is not None
            hostname = properties.find("property[@name='facet_runtime.hostname']")
            assert hostname is not None
            if damage == "testcase-identity":
                case.set("name", case.get("name", "") + "-" + hostname.get("value", ""))
            else:
                case.set("time", hostname.get("value", ""))
        junit = ElementTree.tostring(root)
    result = retain_execution(
        command=["python", "-m", "pytest", _VALIDATOR],
        executed_command=["python", "-m", "pytest", _VALIDATOR],
        repo_root=REPO_ROOT,
        temporary=tmp_path / "temporary",
        stdout=stdout,
        stderr=(actual_validator_archive / "stderr.bin").read_bytes(),
        junit=junit,
        exit_code=0,
        private_directory=tmp_path / "private",
    )
    assert reason in result["provenance_error"]
    assert "junit_xml" not in result
    archive = next((tmp_path / "private").glob("validator-*"))
    assert (archive / "junit.xml").read_bytes() == junit
    assert (archive / "stdout.bin").read_bytes() == stdout
    assert result["raw_junit_sha256"] == hashlib.sha256(junit).hexdigest()


def test_public_projection_keeps_real_results_and_normalises_source_properties(
    tmp_path: Path, actual_validator_archive: Path
) -> None:
    """Keep genuine results while making supplemental source locations portable."""
    import json

    from sc_neurocore.neurons.receipt_execution import junit_checks
    from tools.facet_receipt import REPO_ROOT

    original = (actual_validator_archive / "junit.xml").read_bytes()
    root = ElementTree.fromstring(original)
    properties = root.find(".//properties")
    assert properties is not None
    properties.append(
        ElementTree.Element(
            "property",
            {
                "name": "archive.validator",
                "value": str(REPO_ROOT / _VALIDATOR.split("::")[0]),
            },
        )
    )
    ElementTree.indent(root)
    annotated = ElementTree.tostring(root)
    manifest = json.loads((actual_validator_archive / "manifest.json").read_text())
    result = retain_execution(
        command=manifest["command"],
        executed_command=manifest["executed_command"],
        repo_root=Path(manifest["cwd"]),
        temporary=tmp_path / "temporary",
        stdout=(actual_validator_archive / "stdout.bin").read_bytes(),
        stderr=(actual_validator_archive / "stderr.bin").read_bytes(),
        junit=annotated,
        exit_code=manifest["exit_code"],
        private_directory=tmp_path / "private",
    )
    assert "provenance_error" not in result
    assert junit_checks(result["junit_xml"]) == junit_checks(original.decode())
    projected = ElementTree.fromstring(result["junit_xml"])
    source_property = projected.find(".//property[@name='archive.validator']")
    assert source_property is not None
    assert source_property.get("value") == "<repository>/" + _VALIDATOR.split("::")[0]
    assert [case.attrib for case in root.iter("testcase")] == [
        case.attrib for case in projected.iter("testcase")
    ]
