# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (apply) from former test_readiness_evidence_index.py

from __future__ import annotations

from readiness_evidence_index_support import *  # noqa: F403

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
