# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Criterion→bencher converter contract

"""Contract for the Criterion→bencher converter used by the Performance Benchmarks gate.

The converter feeds ``benchmark-action/github-action-benchmark`` (``fail-on-alert`` at a
500% threshold). A single mis-parsed value poisons the gh-pages baseline and red-gates every
later push, so the parse — especially the unit-boundary straddle that previously under-read
``[999.50 µs 1.0001 ms 1.0050 ms]`` as ``1000 ns`` instead of ``1000100 ns`` — is pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_converter() -> ModuleType:
    """Load the converter script from ``.github`` (it is not an importable package)."""
    path = Path(__file__).resolve().parents[2] / ".github" / "criterion_to_bencher.py"
    spec = importlib.util.spec_from_file_location("criterion_to_bencher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONVERTER = _load_converter()


def _one(line: str) -> int | None:
    """Convert a single Criterion line and return the parsed ns value (or None if skipped)."""
    out = list(_CONVERTER.convert(line))
    if not out:
        return None
    # "test <name> ... bench: <ns> ns/iter (+/- 0)"
    return int(out[0].split("bench:")[1].split("ns/iter")[0].strip())


class TestUnitBoundaryStraddle:
    """The regression the missing test let through: median paired with the wrong unit."""

    def test_us_to_ms_straddle_median_uses_its_own_unit(self) -> None:
        # Median 1.0001 ms must scale as ms (1000100 ns), NOT as the low bound's µs (1000 ns).
        assert _one("vip_1k_steps  time:   [999.50 µs 1.0001 ms 1.0050 ms]") == 1_000_100

    def test_ns_to_us_straddle(self) -> None:
        assert _one("k  time:   [980.0 ns 1.0002 µs 1.0100 µs]") == 1_000  # 1.0002 µs → 1000 ns

    def test_ms_to_s_straddle(self) -> None:
        assert _one("k  time:   [999.0 ms 1.0005 s 1.0100 s]") == 1_000_500_000


class TestSingleUnitEstimates:
    """Non-straddle triplets in each unit convert on the median."""

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("sst_1k_steps  time:   [481.99 µs 482.09 µs 482.91 µs]", 482_090),
            ("adex_1k_steps  time:   [29.9 µs 30.0 µs 30.1 µs]", 30_000),
            ("dense  time:   [25.900 ms 26.075 ms 26.120 ms]", 26_075_000),
            ("fast  time:   [11.0 ns 12.0 ns 13.0 ns]", 12),
            ("slow  time:   [4.9 s 5.0 s 5.1 s]", 5_000_000_000),
        ],
    )
    def test_median_conversion(self, line: str, expected: int) -> None:
        assert _one(line) == expected


class TestNameResolution:
    """The benchmark name comes from the result line or a preceding standalone line."""

    def test_name_on_same_line(self) -> None:
        assert list(_CONVERTER.convert("bench_x  time:   [1.0 µs 2.0 µs 3.0 µs]"))[0].startswith(
            "test bench_x ..."
        )

    def test_standalone_name_line(self) -> None:
        text = "bench_y\n                        time:   [1.0 µs 2.0 µs 3.0 µs]"
        out = list(_CONVERTER.convert(text))
        assert out == ["test bench_y ... bench: 2000 ns/iter (+/- 0)"]

    def test_change_line_does_not_become_name_or_emit(self) -> None:
        # A change: line must neither be captured as a name nor parsed as a measurement.
        text = "bench_z  time:   [1.0 µs 2.0 µs 3.0 µs]\n    change: [-1.0% +0.0% +1.0%]"
        out = list(_CONVERTER.convert(text))
        assert out == ["test bench_z ... bench: 2000 ns/iter (+/- 0)"]


class TestSkippedLines:
    """Lines that are not well-formed measurements yield nothing."""

    def test_time_line_without_bracket_is_skipped(self) -> None:
        assert _one("noisy  time:   pending") is None

    def test_bracket_with_single_estimate_is_skipped(self) -> None:
        assert _one("weird  time:   [42.0 µs]") is None

    def test_warmup_and_progress_lines_are_ignored(self) -> None:
        text = (
            "Benchmarking vip_1k_steps: Warming up for 3.0000 s\n"
            "Benchmarking vip_1k_steps: Collecting 100 samples in estimated 5.01 s\n"
            "vip_1k_steps  time:   [999.50 µs 1.0001 ms 1.0050 ms]\n"
            "Found 3 outliers among 100 measurements (3.00%)"
        )
        assert list(_CONVERTER.convert(text)) == [
            "test vip_1k_steps ... bench: 1000100 ns/iter (+/- 0)"
        ]


def test_main_reads_stdin(monkeypatch, capsys) -> None:
    """The CLI entry point converts stdin to stdout."""
    import io

    monkeypatch.setattr(_CONVERTER.sys, "stdin", io.StringIO("b  time:   [1.0 µs 2.0 µs 3.0 µs]"))
    _CONVERTER.main()
    assert capsys.readouterr().out.strip() == "test b ... bench: 2000 ns/iter (+/- 0)"
