# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark evidence documentation tests

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "benchmarks" / "cross_framework.md"
ARTEFACT_RE = re.compile(r"`((?:benchmarks/results|hdl/reports)/[^`]+)`")


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_cross_framework_evidence_doc_cites_existing_artefacts() -> None:
    text = _doc_text()
    artefacts = sorted(set(ARTEFACT_RE.findall(text)))

    assert artefacts
    for relative_path in artefacts:
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_cross_framework_evidence_doc_tracks_required_frameworks() -> None:
    text = _doc_text()

    for framework in ("Brian2", "snnTorch", "Norse", "NEST", "SpikingJelly"):
        assert framework in text


def test_missing_framework_rows_are_marked_as_gaps() -> None:
    text = _doc_text()

    assert "| NEST | No committed artefact | None | Gap |" in text
    assert "| SpikingJelly | No committed artefact | None | Gap |" in text
    assert "| FPGA power/energy | No committed artefact | None | Gap |" in text
