# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The README is a landing page whose documentation links reach the site

"""Keep the README a landing page whose Documentation links reach pages of the site.

The README is also the package page on PyPI. It keeps what a first reader needs: what the
package is, the quick start, who it is for, the evidence boundary, the generated capability
snapshot, the documentation links, and the community, citation and licence sections. The
reference sections it used to hold (install extras, the Rust engine and benchmark evidence, the
architecture, co-simulation, the example scripts and CI/CD) live on site pages, each linked from
its Documentation section.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_docs_navigation_unique_entries import _navigation, _pages

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
# The sections the landing page keeps measured 299 lines when the reference sections moved to
# the site. A section that would grow the README past this belongs on a site page instead.
MAX_README_LINES = 300
# The site pages that took the README's reference sections, word for word.
MOVED_SECTION_PAGES = frozenset(
    {
        "docs/guides/installation.md",
        "docs/guides/more_ways_in.md",
        "docs/benchmarks/rust_engine_evidence.md",
        "docs/architecture/package_architecture.md",
        "docs/guides/hardware_cosimulation.md",
        "docs/guides/example_scripts.md",
        "docs/development/ci_workflows.md",
    }
)
_LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)")


def _documentation_links() -> list[str]:
    """Return the local link targets of the README's Documentation section.

    Returns
    -------
    list of str
        Repository-relative targets in order; web addresses are left out.
    """
    text = README.read_text(encoding="utf-8")
    section = text.split("\n## Documentation\n", 1)[1].split("\n## ", 1)[0]
    return [
        target
        for target in _LINK.findall(section)
        if not target.startswith(("http://", "https://"))
    ]


def test_the_readme_stays_a_landing_page() -> None:
    """The README, which is also the PyPI page, holds at most ``MAX_README_LINES`` lines."""
    assert len(README.read_text(encoding="utf-8").splitlines()) <= MAX_README_LINES


def test_every_documentation_link_reaches_a_page_of_the_site() -> None:
    """Each local Documentation link names a file; one under ``docs/`` is in the navigation."""
    links = _documentation_links()
    navigation = set(_pages(_navigation()))
    assert links
    for target in links:
        assert (ROOT / target).is_file(), target
        if target.startswith("docs/"):
            assert target.removeprefix("docs/") in navigation, target


def test_every_moved_reference_section_is_linked_from_the_readme() -> None:
    """Each site page that took a README reference section is one of its Documentation links."""
    assert set(_documentation_links()) >= MOVED_SECTION_PAGES
