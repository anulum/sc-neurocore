# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Benchmark evidence documentation tests

from __future__ import annotations

import json
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

    assert "| NEST | No committed artefact | None | Runner available, measurement gap |" in text
    assert (
        "| SpikingJelly | No committed artefact | None | Runner available, measurement gap |"
        in text
    )
    assert (
        "| FPGA power/energy | No committed measurement artefact | "
        "Parser available via `sc-neurocore collect-synthesis` | "
        "Capture path available, measurement gap |"
    ) in text


def test_cross_framework_1k_result_schema_is_current() -> None:
    result_path = REPO_ROOT / "benchmarks" / "results" / "cross_framework_1k.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert isinstance(payload["results"], list)
    assert payload["results"]
    for row in payload["results"]:
        assert isinstance(row["framework"], str)
        assert isinstance(row["mode"], str)
        assert isinstance(row["n_neurons"], int)
        assert isinstance(row["wall_time_s"], (int, float))
        assert isinstance(row["peak_memory_mb"], (int, float))
        assert isinstance(row["n_spikes"], int)
        assert isinstance(row["rate_hz"], (int, float))
