# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WCAG contrast of the Python SVG exporter's text

"""Hold the exported SVG's text to WCAG 2.2 AA.

An exported SVG never reaches a browser this project controls, so no DOM audit
can measure it; the Studio's TypeScript exporter is covered the same way by
``studio/frontend/src/paletteContrast.test.ts``. The arithmetic here is written
out from the specification rather than imported, so a mistake in the product's
own contrast module cannot make this file agree with it.
"""

from __future__ import annotations

import re

import pytest

from sc_neurocore.studio.svg_export import AXIS, BACKGROUND, COLORS, LABEL, traces_to_svg

AA_NORMAL = 4.5
"""Ratio required for text below the large-text size."""

AA_NON_TEXT = 3.0
"""Ratio required for a graphical object needed to understand the content."""

TEXT_FILL = re.compile(r"<text\b[^>]*\bfill=\"(#[0-9a-fA-F]{6})\"")
"""Every ``fill`` on a ``<text>`` element in the exported document."""


def channel(value: int) -> float:
    """Linearise one 8-bit sRGB channel.

    Args:
        value: The channel, 0 to 255.

    Returns:
        The linear-light value used by the relative-luminance sum.
    """
    scaled = value / 255.0
    if scaled <= 0.04045:
        return scaled / 12.92
    return float(((scaled + 0.055) / 1.055) ** 2.4)


def luminance(colour: str) -> float:
    """Compute the relative luminance of a ``#rrggbb`` colour.

    Args:
        colour: A six-digit hexadecimal colour.

    Returns:
        The WCAG relative luminance.
    """
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def contrast(first: str, second: str) -> float:
    """Compute the WCAG contrast ratio between two opaque colours.

    Args:
        first: A six-digit hexadecimal colour.
        second: A six-digit hexadecimal colour.

    Returns:
        The ratio, at least 1.0 and at most 21.0.
    """
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_arithmetic_matches_the_specification() -> None:
    """Check the helpers against the two ratios the specification states."""
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=1e-9)
    assert contrast("#0d1117", "#0d1117") == pytest.approx(1.0, abs=1e-9)
    assert luminance("#808080") == pytest.approx(0.21586, abs=5e-6)


def test_every_text_colour_the_exporter_writes_meets_aa() -> None:
    """Check every ``<text>`` fill in a rendered export against 4.5:1."""
    svg = traces_to_svg(
        [index * 0.1 for index in range(50)],
        {"v": [float(index % 7) for index in range(50)]},
        spikes=[10, 20],
        model_name="LIFNeuron",
    )
    fills = set(TEXT_FILL.findall(svg))
    assert fills, "the exporter wrote no text; the scan no longer matches it"
    failures = {
        fill: round(contrast(fill, BACKGROUND), 2)
        for fill in fills
        if contrast(fill, BACKGROUND) < AA_NORMAL
    }
    assert failures == {}


def test_the_empty_export_states_its_reason_legibly() -> None:
    """Check the no-data export, which takes a different code path."""
    svg = traces_to_svg([], {}, model_name="LIFNeuron")
    fills = set(TEXT_FILL.findall(svg))
    assert fills == {LABEL}
    assert contrast(LABEL, BACKGROUND) >= AA_NORMAL


def test_the_axis_rules_meet_the_non_text_threshold() -> None:
    """Check the axis rules, which are graphics rather than text."""
    assert contrast(AXIS, BACKGROUND) >= AA_NON_TEXT


def test_the_trace_colours_stay_distinguishable_from_the_ground() -> None:
    """Check every trace colour against the non-text threshold."""
    for colour in COLORS:
        assert contrast(colour, BACKGROUND) >= AA_NON_TEXT, colour


def test_the_watermark_is_the_colour_the_typescript_exporter_uses() -> None:
    """Check the value both exporters agree on, so the two cannot drift.

    The TypeScript side reads it from ``PLOT_AXIS`` in
    ``studio/frontend/src/simulationPlotCanvas.ts``; if that value moves, this
    fails and the port is done deliberately rather than by accident.
    """
    assert AXIS == "#727d8b"
