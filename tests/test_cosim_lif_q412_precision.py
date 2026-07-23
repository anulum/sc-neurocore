# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ412Precision from former test_cosim_lif.py

"""Focused suite: TestQ412Precision from former test_cosim_lif.py."""

from __future__ import annotations

from tests.cosim_lif_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ412Precision:
    """Q4.12 precision mode: 4 integer + 12 fractional bits.

    Q4.12 has 1/4096 ≈ 0.00024 resolution (16× finer than Q8.8),
    which dramatically reduces the quantization gap at the cost of
    a narrower integer range ([-8, +7.9997] vs [-128, +127.996]).
    """

    def test_lif_q412_spikes(self) -> None:
        """Q4.12 LIF should spike reliably."""
        vlog_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)
        assert vlog_spikes > 0

    def test_lif_q412_near_python(self) -> None:
        """Q4.12 should close the LIF quantization gap to <5%.

        This is the key precision validation: Q8.8 has a ~99% gap,
        while Q4.12 should be within a few percent of float64.
        """
        py_spikes = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)

        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        print(
            f"\n  Q4.12 co-sim LIF: Python={py_spikes}, Verilog={vlog_spikes}, "
            f"gap={abs(py_spikes - vlog_spikes)} ({gap_pct:.1f}%)"
        )

        # Q4.12 should be within 5% of Python
        assert gap_pct < 5.0, (
            f"Q4.12 gap too large: {gap_pct:.1f}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

    def test_q412_vs_q88_comparison(self) -> None:
        """Compare Q4.12 vs Q8.8 accuracy for LIF.

        With the division fix and look-ahead threshold, both Q8.8 and Q4.12
        achieve near-perfect accuracy for LIF. This test verifies both
        formats are within 5% of Python and documents the comparison.
        """
        py_spikes = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        q88_spikes = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        q412_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)

        gap_q88 = abs(py_spikes - q88_spikes)
        gap_q412 = abs(py_spikes - q412_spikes)

        print(
            f"\n  Precision comparison LIF:"
            f"\n    Q8.8:  Python={py_spikes}, Verilog={q88_spikes}, gap={gap_q88}"
            f"\n    Q4.12: Python={py_spikes}, Verilog={q412_spikes}, gap={gap_q412}"
        )

        # Both should be within 5% of Python
        pct_q88 = gap_q88 / max(py_spikes, 1) * 100
        pct_q412 = gap_q412 / max(py_spikes, 1) * 100
        assert pct_q88 < 5.0, f"Q8.8 gap too large: {pct_q88:.1f}%"
        assert pct_q412 < 5.0, f"Q4.12 gap too large: {pct_q412:.1f}%"

    def test_q412_zero_current_lif_is_range_classified(self) -> None:
        """Q4.12 LIF zero-current is a range mismatch, not a parity claim."""
        params = _lif_schema_precision_values()
        q412 = Q88(data_width=16, fraction=12)
        incompatible = {
            name for name, value in params.items() if not q412.min_value <= value <= q412.max_value
        }
        report = q412.precision_report(dt=1.0, params=params)

        assert q412.min_value == pytest.approx(-8.0)
        assert q412.max_value == pytest.approx(7.999755859375)
        assert incompatible == {"v_rest", "tau_m", "v"}
        assert "Underflow: v_rest=-65.0 below Q4.12 min=-8.0000" in report
        assert "Overflow: tau_m=10.0 exceeds Q4.12 max=7.9998" in report
        assert "Underflow: v=-65.0 below Q4.12 min=-8.0000" in report

        cli = subprocess.run(
            [sys.executable, "-m", "sc_neurocore.neurons", "precision", "lif"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        compatible_line = next(
            line for line in cli.stdout.splitlines() if line.startswith("Compatible modes:")
        )
        assert "Q4.12" not in compatible_line
        assert "Q8.8" in compatible_line
        assert "Q16.16" in compatible_line
