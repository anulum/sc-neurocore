# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (formal_docs) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403


def test_formal_network_verification_docs_cover_cli_and_report_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = repo_root / "docs" / "api" / "formal_network_verification.md"

    text = doc.read_text(encoding="utf-8")

    assert "sc-neurocore formal verify-network" in text
    assert "--refractory-cycles" in text
    assert "--run-symbiyosys" in text
    assert "formal_rate_bound_report.json" in text
    assert "FORMAL_NETWORK_REPORT_SCHEMA_VERSION" in text
    assert "validate_formal_network_report" in text
    assert "tools/verify_formal_network_evidence.py" in text
    assert "formal_network_coverage_manifest.json" in text
    assert "covered_outputs" in text
    assert "artifacts.rtl" in text
    assert "rate_replay" in text
    assert "refractory_replay" in text
    assert "antagonistic_exclusion" in text
    assert "temporal_separation" in text
    assert "population_coactivation" in text
    assert "population_silence" in text
