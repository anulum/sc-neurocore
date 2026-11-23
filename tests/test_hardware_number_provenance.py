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
implementation reports for ``sc_shd_top`` (weightless shell) and
``sc_shd_event_axi`` (with full weight and delay storage in block RAM
and distributed LUTRAM) measured — never a hand-typed figure that drifts
from the artefact, and never an estimator output presented as a
measurement. The ``sc_shd_top`` synthesis excludes the weight ROM and
delay tables, so every quoted figure for it must say it was obtained
without weights or delays, and no page may claim a silicon or board run.
"""

from __future__ import annotations

from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[1]
_UTIL_REPORT = _ROOT / "hdl/reports/vivado_util_xc7z020_100mhz.rpt"
_TIMING_REPORT = _ROOT / "hdl/reports/vivado_timing_xc7z020_100mhz.rpt"
_EVENT_SYNTH_UTIL = _ROOT / "hdl/reports/vivado_sc_shd_event_axi_synth_util_xc7z020_100mhz.rpt"
_EVENT_SYNTH_HIER = _ROOT / "hdl/reports/vivado_sc_shd_event_axi_synth_hier_util_xc7z020_100mhz.rpt"
_EVENT_SYNTH_TIMING = _ROOT / "hdl/reports/vivado_sc_shd_event_axi_synth_timing_xc7z020_100mhz.rpt"
_EVENT_ROUTED_UTIL = _ROOT / "hdl/reports/vivado_sc_shd_event_axi_routed_util_xc7z020_100mhz.rpt"
_EVENT_ROUTED_HIER = (
    _ROOT / "hdl/reports/vivado_sc_shd_event_axi_routed_hier_util_xc7z020_100mhz.rpt"
)
_EVENT_ROUTED_TIMING = (
    _ROOT / "hdl/reports/vivado_sc_shd_event_axi_routed_timing_xc7z020_100mhz.rpt"
)
_EVENT_ROUTED_DRC = _ROOT / "hdl/reports/vivado_sc_shd_event_axi_routed_drc_xc7z020_100mhz.rpt"
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


def test_a_qualified_section_quotes_only_the_reported_figures() -> None:
    """Every LUT and flip-flop count beside the qualifier is the one the report measured.

    The qualifier rule alone accepts a wrong number written next to it, so each
    section or line that carries the qualifier must quote exactly the reported
    LUT and register counts and no other.
    """
    report = _UTIL_REPORT.read_text(encoding="utf-8")
    luts, _ = _site_row(report, "Slice LUTs*")
    regs, _ = _site_row(report, "Slice Registers")
    patterns = {
        luts: (r"\b(\d[\d ,]*)\s?LUT\b", r"\|\s*LUTs\s*\|\s*([\d ,]+)\|"),
        regs: (r"\b(\d[\d ,]*)\s?FF\b", r"\|\s*Flip-flops\s*\|\s*([\d ,]+)\|"),
    }
    checked = 0
    for relative in _PUBLIC_FILES:
        text = (_ROOT / relative).read_text(encoding="utf-8")
        for block in _figure_blocks(relative, text):
            if _QUALIFIER not in block:
                continue
            for expected, expressions in patterns.items():
                for expression in expressions:
                    for figure in re.findall(expression, block):
                        checked += 1
                        assert int(re.sub(r"[ ,]", "", figure)) == expected, (
                            f"{relative} quotes {figure.strip()} beside the qualifier, "
                            f"not the reported {expected}"
                        )
    assert checked >= 8


def _signed_slack(report: str, label: str) -> str:
    """Return the signed slack string (e.g. ``'-4.901'``) for ``Setup`` or ``Hold``."""
    match = re.search(
        rf"^{re.escape(label)}\s*:\s*\d+\s+Failing Endpoints,\s+Worst Slack\s+([-0-9.]+)ns",
        report,
        flags=re.MULTILINE,
    )
    assert match is not None, f"timing report lost its {label!r} worst-slack line"
    return match.group(1)


def test_event_shd_reports_and_public_figures_match_verbatim() -> None:
    """Every public sc_shd_event_axi figure binds to the committed Vivado reports."""
    for path in (
        _EVENT_SYNTH_UTIL,
        _EVENT_SYNTH_HIER,
        _EVENT_SYNTH_TIMING,
        _EVENT_ROUTED_UTIL,
        _EVENT_ROUTED_HIER,
        _EVENT_ROUTED_TIMING,
        _EVENT_ROUTED_DRC,
    ):
        text = path.read_text(encoding="utf-8")
        assert "Design       : sc_shd_event_axi" in text
        assert "| Host         : build-host (" in text
        assert f"hdl/reports/{path.name}" in text

    synth_util = _EVENT_SYNTH_UTIL.read_text(encoding="utf-8")
    s_luts, s_lut_pct = _site_row(synth_util, "Slice LUTs*")
    s_regs, s_reg_pct = _site_row(synth_util, "Slice Registers")
    s_bram, s_bram_pct = _site_row(synth_util, "Block RAM Tile")
    s_ramb36, _ = _site_row(synth_util, "  RAMB36/FIFO*")
    s_ramb18, _ = _site_row(synth_util, "  RAMB18")
    s_dsp, _ = _site_row(synth_util, "DSPs")
    s_wns = _signed_slack(_EVENT_SYNTH_TIMING.read_text(encoding="utf-8"), "Setup")

    routed_util = _EVENT_ROUTED_UTIL.read_text(encoding="utf-8")
    r_luts, r_lut_pct = _site_row(routed_util, "Slice LUTs")
    r_regs, r_reg_pct = _site_row(routed_util, "Slice Registers")
    r_slices, r_slice_pct = _site_row(routed_util, "Slice")
    r_bram, r_bram_pct = _site_row(routed_util, "Block RAM Tile")
    r_dsp, _ = _site_row(routed_util, "DSPs")
    assert (s_bram, r_bram, s_dsp, r_dsp) == (21, 21, 0, 0)

    routed_timing = _EVENT_ROUTED_TIMING.read_text(encoding="utf-8")
    r_wns = _signed_slack(routed_timing, "Setup")
    r_whs = _signed_slack(routed_timing, "Hold")
    min_period_ns = 10.0 - float(r_wns)
    fmax_mhz = f"{1000.0 / min_period_ns:.2f}"

    routed_hier = _EVENT_ROUTED_HIER.read_text(encoding="utf-8")
    assert re.search(
        r"\|\s*\(sc_shd_event_axi\)\s*\|\s*\(top\)\s*\|\s*231\s*\|\s*231\s*\|\s*0\s*\|\s*0\s*\|\s*899\s*\|",
        routed_hier,
    )
    assert re.search(
        r"\|\s*u_core\s*\|\s*sc_shd_event_p\s*\|\s*12375\s*\|\s*12335\s*\|\s*40\s*\|\s*0\s*\|\s*13176\s*\|\s*19\s*\|\s*4\s*\|",
        routed_hier,
    )

    architecture = (_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "| Estimated `Fmax` from `100 MHz` `WNS` |" in architecture
    assert (
        f"| Synthesised | {s_luts:,} ({s_lut_pct:.2f}%) | {s_regs:,} ({s_reg_pct:.2f}%) | "
        f"{s_bram} (`{s_ramb36} RAMB36E1 + {s_ramb18} RAMB18E1`, {s_bram_pct:.2f}%) | "
        f"{s_dsp} (0.00%) | {s_wns} ns | — |"
    ) in architecture
    assert (
        f"| Placed and routed | {r_luts:,} ({r_lut_pct:.2f}%) | {r_regs:,} ({r_reg_pct:.2f}%) | "
        f"{r_bram} (`{s_ramb36} RAMB36E1 + {s_ramb18} RAMB18E1`, {r_bram_pct:.2f}%) | "
        f"{r_dsp} (0.00%) | {r_wns} ns | {fmax_mhz} MHz (`{min_period_ns:.3f} ns`) |"
    ) in architecture
    assert f"`{r_slices:,} / 13,300` slices = `{r_slice_pct:.2f}%`" in architecture
    assert f"`{r_whs} ns`" in architecture
    assert "not measured" in architecture

    derived_label = (
        "estimated from the 100 MHz run's WNS (not a run at that frequency); "
        "hold not closed out of context"
    )
    drc_text = _EVENT_ROUTED_DRC.read_text(encoding="utf-8")
    drc_total_match = re.search(r"Checks found:\s*(\d+)", drc_text)
    assert drc_total_match is not None
    drc_warnings = int(drc_total_match.group(1))
    assert "| Error " not in drc_text
    assert "| REQP-1839 | Warning  | RAMB36 async control check | 20     |" in drc_text
    assert "| REQP-1840 | Warning  | RAMB18 async control check | 20     |" in drc_text
    for drc_page in (
        "ARCHITECTURE.md",
        "ROADMAP.md",
        "docs/benchmarks/BENCHMARKS.md",
        "hdl/README.md",
    ):
        page_text = " ".join((_ROOT / drc_page).read_text(encoding="utf-8").split())
        assert f"`{drc_warnings}` warnings" in page_text
        assert "REQP-1839" in page_text
        assert "REQP-1840" in page_text
        assert derived_label in page_text

    highlight_source = (_ROOT / "tools/architecture_map.toml").read_text(encoding="utf-8")
    assert (
        f"{r_luts:,} LUT ({r_lut_pct:.2f}%), {r_regs:,} FF ({r_reg_pct:.2f}%)" in highlight_source
    )
    assert (
        f"{r_bram} BRAM tiles ({s_ramb36} RAMB36E1 + {s_ramb18} RAMB18E1, {r_bram_pct:.2f}%)"
        in highlight_source
    )
    assert (
        f"WNS {r_wns} ns at 100 MHz ({fmax_mhz} MHz Fmax estimated from the 100 MHz run's WNS; "
        "hold not closed out of context" in highlight_source
    )

    for page in (
        "ROADMAP.md",
        "docs/benchmarks/BENCHMARKS.md",
        "docs/guides/PERFORMANCE_TUNING.md",
        "hdl/README.md",
    ):
        content = " ".join((_ROOT / page).read_text(encoding="utf-8").split())
        assert f"{r_wns} ns" in content
        assert f"{fmax_mhz} MHz" in content
        assert derived_label in content

    tcl = (_ROOT / "hdl/pynq/synth_shd_event_xc7z020.tcl").read_text(encoding="utf-8")
    xdc = (_ROOT / "hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc").read_text(encoding="utf-8")
    assert "read_xdc ${repo_root}/hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc" in tcl
    assert "create_clock -period 10.000 -name clk [get_ports S_AXI_ACLK]" in xdc
