# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Power estimation

"""Power estimation utilities.

Switching-activity-based dynamic/static power model from generated Verilog
without synthesis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PowerEstimate:
    """Estimated power consumption for a compiled neuron.

    Attributes
    ----------
    dynamic_mw : float
        Dynamic (switching) power in milliwatts.
    static_mw : float
        Leakage power in milliwatts.
    total_mw : float
        Total power (dynamic + static).
    energy_per_spike_nj : float
        Energy per spike event in nanojoules.
    toggle_rate : float
        Average toggle rate (transitions per clock per bit).
    """

    dynamic_mw: float
    static_mw: float
    total_mw: float
    energy_per_spike_nj: float
    toggle_rate: float


def _load_vcd_text(activity_vcd: str | Path) -> str:
    if isinstance(activity_vcd, Path):
        return activity_vcd.read_text(encoding="utf-8")
    candidate = Path(activity_vcd)
    if "$var" not in activity_vcd and candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return activity_vcd


def _bit_toggle_count(previous: str, current: str, width: int) -> int:
    previous = previous[-width:].zfill(width)
    current = current[-width:].zfill(width)
    return sum(a != b for a, b in zip(previous, current, strict=True))


def _parse_vcd_activity(
    activity_vcd: str | Path, time_units_per_cycle: float
) -> tuple[float, float]:
    """Return measured toggles per cycle and average bit toggle rate from VCD."""
    if time_units_per_cycle <= 0 or not math.isfinite(time_units_per_cycle):
        raise ValueError("vcd_time_units_per_cycle must be finite and > 0")

    text = _load_vcd_text(activity_vcd)
    widths: dict[str, int] = {}
    previous_values: dict[str, str] = {}
    observed_codes: set[str] = set()
    current_time = 0.0
    first_time: float | None = None
    last_time: float | None = None
    total_toggles = 0

    var_re = re.compile(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+.+?\$end")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        var_match = var_re.match(line)
        if var_match:
            widths[var_match.group(2)] = int(var_match.group(1))
            continue
        if line.startswith("#"):
            current_time = float(line[1:])
            if first_time is None:
                first_time = current_time
            last_time = current_time
            continue
        if line[0] in "01" and len(line) >= 2:
            value = line[0]
            code = line[1:].strip()
        elif line[0] in "bB":
            parts = line.split()
            if len(parts) != 2:
                continue
            value = parts[0][1:]
            code = parts[1]
        else:
            continue
        if code not in widths or any(bit not in "01" for bit in value):
            continue
        observed_codes.add(code)
        if code in previous_values:
            total_toggles += _bit_toggle_count(previous_values[code], value, widths[code])
        previous_values[code] = value

    if not observed_codes:
        return 0.0, 0.0
    if first_time is None or last_time is None or last_time <= first_time:
        cycles = 1.0
    else:
        cycles = max(1.0, (last_time - first_time) / time_units_per_cycle)
    observed_bits = sum(widths[code] for code in observed_codes)
    toggles_per_cycle = total_toggles / cycles
    toggle_rate = total_toggles / max(1.0, observed_bits * cycles)
    return toggles_per_cycle, toggle_rate


def estimate_power(
    verilog: str,
    *,
    data_width: int = 16,
    freq_mhz: float = 100.0,
    vdd: float = 1.0,
    process_nm: int = 28,
    spike_rate_hz: float = 10.0,
    activity_vcd: str | Path | None = None,
    vcd_time_units_per_cycle: float = 1.0,
) -> PowerEstimate:
    """Estimate power consumption from generated Verilog.

    When a VCD trace is provided, dynamic power uses measured bit-level
    switching activity. Otherwise the function falls back to structural
    activity factors derived from registers, adders, multipliers, and
    technology parameters.

    Parameters
    ----------
    verilog : str
        Generated Verilog source.
    data_width : int
        Fixed-point data width.
    freq_mhz : float
        Clock frequency in MHz.
    vdd : float
        Supply voltage (V).
    process_nm : int
        Process node in nm (7, 16, 28, 45, ...).
    spike_rate_hz : float
        Expected average spike rate.
    activity_vcd : str or Path, optional
        VCD trace text or a path to a VCD file. When provided, bit-level
        transitions in the trace drive dynamic power.
    vcd_time_units_per_cycle : float
        Number of VCD timestamp units per target clock cycle.

    Returns
    -------
    PowerEstimate
        Estimated power breakdown.
    """
    # Count switching elements
    mul_count = len(re.findall(r"wire\s+signed\s+\[.*?\]\s+_mul\d+", verilog))
    add_count = verilog.count(" + ") + verilog.count(" - ")
    reg_count = len(re.findall(r"reg\s+signed\s+\[", verilog))

    # Technology-dependent capacitance (fF per toggle per bit)
    cap_per_bit_ff = {
        7: 0.2,
        10: 0.3,
        14: 0.5,
        16: 0.6,
        22: 0.8,
        28: 1.0,
        40: 1.5,
        45: 1.8,
        65: 2.5,
    }.get(process_nm, 1.0)

    if activity_vcd is not None:
        total_toggles, avg_toggle_rate = _parse_vcd_activity(
            activity_vcd,
            vcd_time_units_per_cycle,
        )
    else:
        # Structural activity factors used when measured switching data is absent.
        reg_toggles = reg_count * data_width * 0.25
        add_toggles = add_count * data_width * 0.50
        mul_toggles = mul_count * data_width * data_width * 0.10

        total_toggles = reg_toggles + add_toggles + mul_toggles
        avg_toggle_rate = total_toggles / max(1, (reg_count + add_count + mul_count) * data_width)

    # P_dynamic = α × C × V² × f
    cap_f = cap_per_bit_ff * 1e-15  # convert fF to F
    freq_hz = freq_mhz * 1e6
    p_dynamic_w = total_toggles * cap_f * (vdd**2) * freq_hz
    p_dynamic_mw = p_dynamic_w * 1e3

    # Leakage: ~0.1 μW per LUT for 28nm (scales with process²)
    leakage_uw_per_lut = 0.1 * (process_nm / 28) ** 2
    lut_estimate = add_count * data_width + mul_count * data_width * data_width // 4
    p_static_mw = lut_estimate * leakage_uw_per_lut * 1e-3

    total_mw = p_dynamic_mw + p_static_mw

    # Energy per spike (nJ) = total_power / spike_rate
    if spike_rate_hz > 0:
        energy_per_spike_nj = (total_mw * 1e-3) / spike_rate_hz * 1e9
    else:
        energy_per_spike_nj = 0.0

    return PowerEstimate(
        dynamic_mw=round(p_dynamic_mw, 4),
        static_mw=round(p_static_mw, 4),
        total_mw=round(total_mw, 4),
        energy_per_spike_nj=round(energy_per_spike_nj, 2),
        toggle_rate=round(avg_toggle_rate, 3),
    )
