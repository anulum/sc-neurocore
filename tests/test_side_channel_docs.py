# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Side-channel documentation boundary tests

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "guides" / "side_channel_encoding.md"
SECURITY_API = REPO_ROOT / "docs" / "api" / "security.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"


def test_side_channel_guide_documents_evidence_boundary_and_non_claims() -> None:
    text = GUIDE.read_text(encoding="utf-8")

    assert "analytic_simulation_only" in text
    assert "no physical power measurement" in text
    assert "no physical thermal measurement" in text
    assert "no DPA-resistance claim" in text
    assert "no silicon-security claim" in text
    assert "side_channel_benchmark.py" in text
    assert "side_channel_hdl_emit.py" in text
    assert "deploy_manifest" in text
    assert "side-channel security certification" not in text.lower()


def test_side_channel_docs_are_linked_from_nav_and_security_api() -> None:
    nav = MKDOCS.read_text(encoding="utf-8")
    api = SECURITY_API.read_text(encoding="utf-8")

    assert "Side-Channel Encoding: guides/side_channel_encoding.md" in nav
    assert "sc_neurocore.security.side_channel_metrics" in api
    assert "sc_neurocore.security.thermal_sc_encoding" in api
    assert "sc_neurocore.security.side_channel_benchmark" in api
