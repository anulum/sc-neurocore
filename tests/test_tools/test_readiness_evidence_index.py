# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — readiness_evidence_index tool tests

"""Real-surface tests for tools/readiness_evidence_index.py.

Exercises the shipped module (not a reimplementation): inventory construction,
facet builders, apply dry-path against a temporary descriptor payload, and the
CLI entry points.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest
import tomli_w

from sc_neurocore.neurons.model_descriptor import ModelDescriptor

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "readiness_evidence_index.py"


def _load_tool() -> ModuleType:
    """Load the readiness evidence index tool as a real module from disk."""
    import sys

    name = "readiness_evidence_index_under_test"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses with slots require the module to be registered first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Shared loaded tool module."""
    return _load_tool()


def test_enrolled_shortlist_excludes_peer_wang_buzsaki_from_apply(tool: ModuleType) -> None:
    """Wang-Buzsaki remains inventored but skip_apply for peer-lane isolation."""
    wb = [e for e in tool.ENROLLED if e.schema_name == "wang_buzsaki"]
    assert len(wb) == 1
    assert wb[0].skip_apply is True
    assert "peer" in wb[0].skip_reason.lower() or "Gauss" in wb[0].skip_reason


def test_enrolled_class_names_exist_in_descriptor_corpus(tool: ModuleType) -> None:
    """Every enrolled class_name has an on-disk descriptor (except none expected)."""
    from sc_neurocore.neurons.model_catalogue import descriptor_path

    missing = [e.class_name for e in tool.ENROLLED if not descriptor_path(e.class_name).is_file()]
    assert missing == [], f"missing descriptors: {missing}"


def test_expif_is_enrolled_at_q3232_cosim_tier(tool: ModuleType) -> None:
    """ExpIF no longer inherits the obsolete compile-only schema-gap claim."""
    expif = next(e for e in tool.ENROLLED if e.class_name == "ExpIFNeuron")
    assert expif.level == "h1_cosim"
    assert expif.evidence == "tests/test_cosim_exp_if.py::test_expif_q3232_spike_parity"
    assert "Q32.32" in expif.operating_point
    assert "0/0/1/2/5/9" in expif.tolerance


def test_lapicque_is_enrolled_with_dedicated_exact_flow_evidence(tool: ModuleType) -> None:
    """Replace the generic suite pointer with the measured Lapicque contract."""
    lapicque = next(e for e in tool.ENROLLED if e.class_name == "LapicqueNeuron")
    assert lapicque.level == "h1_cosim"
    assert lapicque.evidence == (
        "tests/test_cosim_lapicque.py::test_q1616_preserves_event_vectors_and_voltage_bound"
    )
    assert "I=0.333,2.3,20.25" in lapicque.operating_point
    assert "Q16.16 event vectors exact" in lapicque.tolerance


def test_quadratic_if_is_enrolled_with_dedicated_exact_flow_evidence(
    tool: ModuleType,
) -> None:
    """Replace the removed generic-suite pointer with measured QIF evidence."""
    quadratic_if = next(e for e in tool.ENROLLED if e.class_name == "QuadraticIFNeuron")
    assert quadratic_if.level == "h1_cosim"
    assert quadratic_if.evidence == (
        "tests/test_cosim_quadratic_if.py::test_q1616_preserves_event_vectors_and_voltage_bound"
    )
    assert "I=0,0.333,0.5,1,2,5,20,50" in quadratic_if.operating_point
    assert "Q16.16 event vectors exact" in quadratic_if.tolerance


def test_theta_is_enrolled_with_dedicated_exact_flow_evidence(
    tool: ModuleType,
) -> None:
    """Replace the generic transcendental-suite pointer with measured Theta evidence."""
    theta = next(e for e in tool.ENROLLED if e.class_name == "ThetaNeuron")
    assert theta.level == "h1_cosim"
    assert theta.evidence == (
        "tests/test_cosim_theta.py::test_q1616_preserves_complete_event_count_vector"
    )
    assert "I=-1,-0.5,0,0.1,0.333,0.5,1,2,5,20,50" in theta.operating_point
    assert "Q16.16 event counts exact" in theta.tolerance
    assert "below 0.17 rad" in theta.tolerance


def test_build_rows_returns_one_row_per_enrolled(tool: ModuleType) -> None:
    """Inventory length matches the curated enrolment table."""
    rows = tool.build_rows()
    assert len(rows) == len(tool.ENROLLED)
    names = {row.class_name for row in rows}
    assert "AdExNeuron" in names
    assert "WangBuzsakiNeuron" in names


def test_validation_and_silicon_sections_are_honest_for_h1(tool: ModuleType) -> None:
    """H1 entries claim parity metric and cosim anchors; H0 does not claim cosim."""
    h1 = next(e for e in tool.ENROLLED if e.level == "h1_cosim" and not e.skip_apply)
    h0 = next(e for e in tool.ENROLLED if e.level == "h0_compile")
    v1 = tool.validation_section(h1, has_dynamics=True)
    s1 = tool.silicon_section(h1)
    assert v1["dynamics_faithful"] is True
    assert v1["metric"] == "parity"
    assert v1["evidence"]
    assert s1["compiles"] is True
    assert s1["cosim_validated"] is True
    assert s1["target_tier"] == "H1"

    v0 = tool.validation_section(h0, has_dynamics=True)
    s0 = tool.silicon_section(h0)
    assert v0["metric"] == "none"
    assert s0["compiles"] is True
    assert s0["cosim_validated"] is False
    assert s0["target_tier"] == "H0"
    v0_empty = tool.validation_section(h0, has_dynamics=False)
    assert v0_empty["dynamics_faithful"] is False


def test_index_payload_schema_and_json_roundtrip(tool: ModuleType, tmp_path: Path) -> None:
    """JSON export is valid and carries the schema version marker."""
    rows = tool.build_rows()
    payload = tool.index_payload(rows)
    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["enrolled_count"] == len(tool.ENROLLED)
    out = tmp_path / "index.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == tool.SCHEMA_VERSION
    assert len(loaded["rows"]) == len(rows)


def test_main_report_exits_zero(tool: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --report runs against the live corpus without error."""
    code = tool.main(["--report"])
    captured = capsys.readouterr()
    assert code == 0
    assert "AdExNeuron" in captured.out
    assert "open_gaps=" in captured.out


def test_apply_writes_facets_and_raises_live_tiers(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--apply path: write facets for one enrolled model via temp descriptor dir.

    Uses the real AdExNeuron payload as a template, redirects descriptor I/O to a
    temporary directory so the WC-A5 peer tree is not mutated by this unit test.
    """
    from sc_neurocore.neurons import model_catalogue as catalogue

    src_path = catalogue.descriptor_path("AdExNeuron")
    assert src_path.is_file()
    tmp_desc = tmp_path / "AdExNeuron.toml"

    raw = src_path.read_text(encoding="utf-8")
    body = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
    initial_payload = tomllib.loads(body)
    initial_payload.pop("validation", None)
    initial_payload.pop("silicon", None)
    tmp_desc.write_text(
        tool._DESCRIPTOR_HEADER + tomli_w.dumps(initial_payload),
        encoding="utf-8",
    )

    def _fake_path(class_name: str) -> Path:
        if class_name == "AdExNeuron":
            return tmp_desc
        return catalogue.DESCRIPTOR_DIR / f"{class_name}.toml"

    monkeypatch.setattr(tool, "descriptor_path", _fake_path)
    # load_descriptor_payload and load_descriptor use model_catalogue paths —
    # patch those to the temp file for AdExNeuron only.
    real_load_payload = catalogue.load_descriptor_payload
    real_load = catalogue.load_descriptor

    def _payload(class_name: str) -> dict[str, Any] | None:
        if class_name == "AdExNeuron":
            raw = tmp_desc.read_text(encoding="utf-8")
            body = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
            return tomllib.loads(body)
        return real_load_payload(class_name)

    def _load(class_name: str) -> ModelDescriptor | None:
        if class_name == "AdExNeuron":
            from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

            payload = _payload(class_name)
            assert payload is not None
            return parse_model_descriptor(payload)
        return real_load(class_name)

    monkeypatch.setattr(tool, "load_descriptor_payload", _payload)
    monkeypatch.setattr(tool, "load_descriptor", _load)

    entry = next(e for e in tool.ENROLLED if e.class_name == "AdExNeuron")
    lines = tool.apply_facets((entry,))
    assert any(line.startswith("APPLIED AdExNeuron") for line in lines)
    text = tmp_desc.read_text(encoding="utf-8")
    assert "dynamics_faithful" in text
    assert "compiles" in text
    desc = _load("AdExNeuron")
    assert desc is not None
    # Silicon climbs on compile/cosim anchors alone; science stays at the S0–S3
    # kernel until multi-backend + reproducibility curation reaches S3.
    assert tool.silicon_tier(desc) == 1
    assert desc.validation.dynamics_faithful is True
    assert desc.silicon.compiles is True
    assert desc.silicon.cosim_validated is True


def test_apply_preserves_stronger_rulkov_facets(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Applying an H0 floor must not erase an existing trajectory-backed H2 descriptor."""
    from sc_neurocore.neurons import model_catalogue as catalogue

    source = catalogue.descriptor_path("RulkovMapNeuron")
    temporary = tmp_path / "RulkovMapNeuron.toml"
    temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    before = temporary.read_bytes()

    def _path(class_name: str) -> Path:
        return (
            temporary if class_name == "RulkovMapNeuron" else catalogue.descriptor_path(class_name)
        )

    def _payload(class_name: str) -> dict[str, Any] | None:
        if class_name == "RulkovMapNeuron":
            raw = temporary.read_text(encoding="utf-8")
            body = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
            return tomllib.loads(body)
        return catalogue.load_descriptor_payload(class_name)

    def _load(class_name: str) -> ModelDescriptor | None:
        if class_name == "RulkovMapNeuron":
            from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

            payload = _payload(class_name)
            assert payload is not None
            return parse_model_descriptor(payload)
        return catalogue.load_descriptor(class_name)

    monkeypatch.setattr(tool, "descriptor_path", _path)
    monkeypatch.setattr(tool, "load_descriptor_payload", _payload)
    monkeypatch.setattr(tool, "load_descriptor", _load)

    entry = next(item for item in tool.ENROLLED if item.class_name == "RulkovMapNeuron")
    lines = tool.apply_facets((entry,))

    assert lines == ["PRESERVED RulkovMapNeuron: existing S5 H2 meets or exceeds h0_compile"]
    assert temporary.read_bytes() == before


def test_apply_reports_descriptor_reload_failure(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facet application fails closed when the descriptor cannot be parsed."""

    def _missing_descriptor(_class_name: str) -> None:
        return None

    monkeypatch.setattr(tool, "load_descriptor", _missing_descriptor)
    entry = next(item for item in tool.ENROLLED if item.class_name == "RulkovMapNeuron")

    assert tool.apply_facets((entry,)) == ["MISS RulkovMapNeuron: descriptor reload failed"]
