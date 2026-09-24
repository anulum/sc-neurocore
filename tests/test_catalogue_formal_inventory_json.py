# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The formal inventory states what each job checks and leaves open

"""``inventory.json`` describes the committed jobs as they are, not as claimed."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "hdl" / "formal" / "catalogue"


def _inventory() -> dict[str, object]:
    data = json.loads((CATALOGUE / "inventory.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _jobs() -> list[dict[str, object]]:
    data = _inventory()
    jobs = data["jobs"]
    retained = data["retained_jobs"]
    assert isinstance(jobs, list) and isinstance(retained, list)
    return [*jobs, *retained]


def test_every_job_is_a_bounded_safety_check_with_its_limits_stated() -> None:
    data = _inventory()
    assert data["schema_version"] == "sc-neurocore.formal-inventory.v1"
    assert "not verified" in str(data["enrolment"])
    for job in _jobs():
        assert job["claim"] == "bounded safety"
        assert job["mode"] == "bmc"
        not_established = job["not_established"]
        assert isinstance(not_established, list)
        assert "equivalence with the model or with the bit-true kernel" in not_established
        assert f"any behaviour after {job['depth']} cycles" in not_established
        assert job["properties"] and job["assumptions"]
        if job["origin"] == "curated":
            assert any("committed RTL file" in item for item in not_established)


def test_each_job_names_committed_artefacts_and_its_reachability_honestly() -> None:
    for job in _jobs():
        module = str(job["module"])
        harness = (CATALOGUE / f"{module}_formal.v").read_text(encoding="utf-8")
        sby = (CATALOGUE / f"{module}.sby").read_text(encoding="utf-8")
        assert (CATALOGUE / f"{module}.v").is_file()
        assert f"depth {job['depth']}" in sby
        assert f"smtbmc {job['solver']}" in sby
        assert job["reachability_asserted"] == ("assert (seen_spike)" in harness)
        if not job["reachability_asserted"]:
            reasons = job["not_established"]
            assert isinstance(reasons, list)
            assert any("vacuously" in str(item) for item in reasons)


def test_no_harness_asserts_that_a_word_lies_in_its_own_range() -> None:
    """``$signed(x) >= -2**(w-1)`` on a signed w-bit port cannot fail."""
    tautology = re.compile(r">= -(\d+)'sd(\d+)\);")
    for harness in sorted(CATALOGUE.glob("*_formal.v")):
        for width, magnitude in tautology.findall(harness.read_text(encoding="utf-8")):
            assert int(magnitude) != 1 << (int(width) - 1), harness.name


def test_the_markdown_inventory_and_the_json_list_the_same_jobs() -> None:
    markdown = (CATALOGUE / "INVENTORY.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| (\w+) \| \w+ \| `(\w+)`", markdown, re.MULTILINE))
    assert listed == {(str(job["class"]), str(job["module"])) for job in _jobs()}
    assert all(job["class"] != "McKeanNeuron" for job in _jobs())
