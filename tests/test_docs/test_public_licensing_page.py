# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - public licensing docs test

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LICENSE_DOC = REPO_ROOT / "docs" / "development" / "licensing.md"


def _lowered_content(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_public_licensing_page_exists() -> None:
    assert LICENSE_DOC.exists(), f"missing licensing page: {LICENSE_DOC.relative_to(REPO_ROOT)}"


def test_public_licensing_page_contains_required_wording() -> None:
    text = _lowered_content(LICENSE_DOC)

    required_fragments = [
        "agpl-3.0-or-later",
        "commercial licensing",
        "commercial licensing is available",
        "modified server deployments",
        "network-service",
        "source-availability",
        "modified code must be made available",
        "separate commercial agreement",
        "model and data artefact matrix",
        "artefact-specific licence",
        "pretrained artefacts",
        "model weights",
        "datasets",
        "security/model_data_license_matrix.json",
    ]

    for phrase in required_fragments:
        assert phrase in text, f"missing required phrase: {phrase!r}"


def test_public_licensing_page_has_no_internal_vendor_names() -> None:
    text = _lowered_content(LICENSE_DOC)
    forbidden = [
        "co" + "dex",
        "open" + "ai",
        "cla" + "ude",
        "anth" + "ropic",
        "gem" + "ini",
        "g" + "pt",
    ]

    for term in forbidden:
        assert term not in text, f"forbidden term in public licensing page: {term}"
