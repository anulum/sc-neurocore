# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — analysis toolkit function count on the public pages

"""Bind every public statement of the analysis toolkit's function count to the package's exports.

The pages once said 124, 125, 127 and 132. The count is the number of functions in
``sc_neurocore.analysis.__all__``; each current page that states it is listed here, and a page
that states it without being listed fails. Two dated records keep the counts of their time.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import sc_neurocore.analysis as analysis

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = sum(inspect.isfunction(getattr(analysis, name)) for name in analysis.__all__)

PAGES = {
    "docs/api/analysis.md": (
        r"^# Spike Train Analysis -- (\d+) Functions$",
        r"exports (\d+) functions and one class",
    ),
    "docs/architecture/SYSTEM_MAP.md": (r"spike-train toolkit, (\d+) functions",),
    "docs/COMPETITIVE_LANDSCAPE.md": (r"\| Spike train analysis \| \*\*(\d+) functions\*\*",),
    "docs/guides/FOR_RESEARCH_LABS.md": (r"^(\d+) analysis functions, zero external dependencies",),
    "docs/guides/notebook_guide.md": (r"van Rossum, PCA \((\d+) functions\)",),
    "docs/tutorials/29_spikeinterface_import.md": (
        r"Spike Train Analysis\]\([^)]+\) — (\d+) analysis",
    ),
}
# Dated records: the v3.12 milestone list and the v3.13 capability report state the count of their
# release, and are not rewritten.
DATED_RECORDS = {
    "ROADMAP.md",
    "docs/reports/SC_NEUROCORE_CAPABILITY_REPORT_v3.13.md",
}
COUNT_STATEMENT = re.compile(
    r"\b\d+ analysis functions\b|[Ss]pike[- ]train (?:analysis|toolkit)[^\n]{0,40}?\b\d+ functions"
    r"|\bSpike Train Analysis -- \d+ Functions",
)


def test_the_count_is_the_packages_exported_functions() -> None:
    """The exports hold 127 functions and the one class the analysis page names."""
    classes = [name for name in analysis.__all__ if inspect.isclass(getattr(analysis, name))]
    assert classes == ["SpikeToConceptMapper"]
    assert FUNCTIONS == len(analysis.__all__) - 1 == 127


@pytest.mark.parametrize(("page", "patterns"), sorted(PAGES.items()))
def test_each_current_page_states_the_exported_count(page: str, patterns: tuple[str, ...]) -> None:
    """Every statement of the count on a current page equals the number of exported functions."""
    text = (ROOT / page).read_text(encoding="utf-8")
    for pattern in patterns:
        found = re.findall(pattern, text, flags=re.MULTILINE)
        assert found == [str(FUNCTIONS)], (page, pattern, found)


def test_no_other_page_states_the_count() -> None:
    """A page that states the count must be listed above, so that it is bound to the exports."""
    stating = {
        path.relative_to(ROOT).as_posix()
        for path in [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")]
        if "CHANGELOG" not in path.name
        and COUNT_STATEMENT.search(path.read_text(encoding="utf-8", errors="replace"))
    }
    assert stating, "the pattern matches no page, so it no longer finds a statement"
    assert stating <= set(PAGES) | DATED_RECORDS, sorted(stating - set(PAGES) - DATED_RECORDS)


def test_the_analysis_page_claims_no_completeness_against_other_libraries() -> None:
    """The page no longer says it covers every metric of other toolkits, which no test checks."""
    text = (ROOT / "docs/api/analysis.md").read_text(encoding="utf-8")
    assert "every metric" not in text
    assert "claims neither completeness nor numerical agreement" in text
