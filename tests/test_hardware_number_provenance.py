# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public hardware numbers bind to committed reports

"""Bind every public SHD hardware number to its committed Vivado report.

The public architecture pages and the generated capability highlights
must quote exactly what the committed out-of-context synthesis and
timing reports for ``sc_shd_top`` measured — never a hand-typed figure
that drifts from the artefact, and never an estimator output presented
as a measurement. That synthesis excludes the weight ROM and delay
tables, so every quoted figure must say it was obtained without weights
or delays, and no page may claim a silicon run.
"""

from __future__ import annotations

from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[1]
_UTIL_REPORT = _ROOT / "hdl/reports/vivado_util_xc7z020_100mhz.rpt"
_TIMING_REPORT = _ROOT / "hdl/reports/vivado_timing_xc7z020_100mhz.rpt"
_PUBLIC_FILES = (
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "tools/architecture_map.toml",
    "README.md",
    "docs/index.md",
    "docs/architecture/SYSTEM_MAP.md",
    "docs/benchmarks/BENCHMARKS.md",
    "docs/COMPETITIVE_LANDSCAPE.md",
    "docs/_generated/capability_manifest.json",
    "docs/_generated/studio_manifest.json",
)
_QUALIFIER = "without weights or delays"
_SILICON_CLAIMS = (
    "proven on silicon",
    "proven end-to-end",
    "silicon-proven",
    "real Xilinx silicon",
    "fits in under 2%",
)


def _site_row(report: str, site: str) -> tuple[int, float]:
    """Return (used, util%) for one site-type row of the utilisation report."""
    pattern = rf"^\| {re.escape(site)}\s*\|\s*(\d+)\s\|.*\|\s*([0-9.]+)\s\|$"
    match = re.search(pattern, report, flags=re.MULTILINE)
    assert match is not None, f"utilisation report lost its {site!r} row"
    return int(match.group(1)), float(match.group(2))


def test_architecture_table_matches_the_committed_reports() -> None:
    """The ARCHITECTURE.md table quotes the artefact values verbatim."""
    report = _UTIL_REPORT.read_text(encoding="utf-8")
    luts, lut_pct = _site_row(report, "Slice LUTs*")
    regs, reg_pct = _site_row(report, "Slice Registers")
    assert "Design       : sc_shd_top" in report

    timing = _TIMING_REPORT.read_text(encoding="utf-8")
    wns_match = re.search(r"Worst Slack\s+([0-9.]+)ns", timing)
    assert wns_match is not None, "timing report lost its worst-slack summary"
    wns = wns_match.group(1)

    architecture = (_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert f"| LUTs | {luts} | 53,200 | {lut_pct}% |" in architecture
    assert f"| Flip-flops | {regs} | 106,400 | {reg_pct}% |" in architecture
    assert f"WNS | +{wns} ns" in architecture
    assert f"{luts} LUT ({lut_pct}%), {regs} FF ({reg_pct}%), WNS +{wns} ns" in architecture
    assert "hdl/reports/vivado_util_xc7z020_100mhz.rpt" in architecture


def test_capability_highlight_matches_the_committed_report() -> None:
    """The generated-manifest source quotes the artefact LUT count."""
    report = _UTIL_REPORT.read_text(encoding="utf-8")
    luts, _ = _site_row(report, "Slice LUTs*")
    highlight_source = (_ROOT / "tools/architecture_map.toml").read_text(encoding="utf-8")
    assert f"{luts} LUT" in highlight_source


def test_public_pages_do_not_carry_drifted_or_estimated_hardware_numbers() -> None:
    """Retired hand-typed figures and estimator outputs stay unpublished."""
    for relative in _PUBLIC_FILES:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        for stale in ("1317 LUT", "1 317", "1,317", "848 FF", "56,243", "56243"):
            assert stale not in text, f"{relative} republishes the drifted figure {stale!r}"
        for claim in _SILICON_CLAIMS:
            assert claim not in text, (
                f"{relative} claims {claim!r} while no board run or weighted synthesis exists"
            )


def _figure_blocks(relative: str, text: str) -> list[str]:
    """Split a public file into the blocks that must carry the qualifier themselves.

    Markdown pages are split at headings, so a section that quotes a figure must
    qualify it; any other file is split into lines, one setting or string each.
    """
    if relative.endswith(".md"):
        return re.split(r"^#{1,6} ", text, flags=re.MULTILINE)
    return text.splitlines()


def test_every_quoted_synthesis_figure_says_it_has_no_weights() -> None:
    """A section or line that quotes the report's LUT count also names what it omits."""
    luts, _ = _site_row(_UTIL_REPORT.read_text(encoding="utf-8"), "Slice LUTs*")
    figure = re.compile(rf"\b{luts} LUT|\|\s*LUTs\s*\|\s*{luts}\s*\|")
    quoted = 0
    for relative in _PUBLIC_FILES:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        for block in _figure_blocks(relative, text):
            if figure.search(block):
                quoted += 1
                assert _QUALIFIER in block, (
                    f"{relative} quotes the {luts} LUT synthesis without saying it has no weights"
                )
    assert quoted >= 5
