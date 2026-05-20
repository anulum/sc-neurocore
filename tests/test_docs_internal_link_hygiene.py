# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path


PUBLIC_DOCS_WITH_PRIOR_EXCLUDED_LINK_WARNINGS = (
    "docs/api/rust-analysis-engine.md",
    "docs/guides/cosimulation_guide.md",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_docs_do_not_link_excluded_internal_pages() -> None:
    root = _repo_root()

    for rel_path in PUBLIC_DOCS_WITH_PRIOR_EXCLUDED_LINK_WARNINGS:
        text = (root / rel_path).read_text(encoding="utf-8")

        assert "../internal/" not in text, rel_path
        assert "](internal/" not in text, rel_path
