# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_quantum_cognition_docs_keep_internal_brain_backlog_private() -> None:
    text = (_repo_root() / "docs" / "api" / "quantum_cognition.md").read_text(encoding="utf-8")

    assert "TODO_GOTM_BRAIN" not in text
    assert "docs/internal" not in text
    assert "../internal/" not in text


def test_gotm_brain_public_docs_remain_explicitly_experimental() -> None:
    text = (_repo_root() / "docs" / "api" / "quantum_cognition.md").read_text(encoding="utf-8")
    section = text.split("### 10.3 GOTM Brain integration", maxsplit=1)[1].split(
        "### 10.4 Population integration", maxsplit=1
    )[0]
    section_lower = section.lower()

    assert "experimental" in section_lower
    assert "not hardware validation evidence" in section_lower
    assert "not a biological claim" in section_lower
    assert "reproducible run packs" in section_lower
