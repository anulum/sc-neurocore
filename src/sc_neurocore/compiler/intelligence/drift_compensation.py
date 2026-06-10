# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analog drift compensation

"""Analog drift compensation for memristive and analog CIM targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftCompensator:
    """Analog drift compensation parameters.

    Attributes
    ----------
    refresh_interval_ms : float
        How often to re-calibrate (ms).
    drift_rate_per_day : float
        Expected weight drift per day.
    compensation_method : str
        ``"periodic_refresh"``, ``"adaptive"``, or ``"ecc"``.
    verilog_controller : str
        Generated Verilog refresh controller.
    """

    refresh_interval_ms: float
    drift_rate_per_day: float
    compensation_method: str
    verilog_controller: str


def generate_drift_compensator(
    module_name: str,
    *,
    drift_rate_per_day: float = 0.001,
    max_drift_tolerance: float = 0.05,
    clock_freq_mhz: int = 100,
    compensation_method: str = "periodic_refresh",
) -> DriftCompensator:
    """Generate analog drift compensation controller."""
    if drift_rate_per_day > 0:
        days_to_tolerance = max_drift_tolerance / drift_rate_per_day
        refresh_ms = days_to_tolerance * 24 * 3600 * 1000
    else:
        refresh_ms = 1e9

    cycles = int(refresh_ms * clock_freq_mhz * 1000)

    v = [
        f"// Drift compensation controller for {module_name}",
        f"// SC-NeuroCore — {compensation_method} method",
        f"// Refresh every {refresh_ms:.0f} ms ({cycles} cycles)",
        "",
        f"module {module_name}_drift_ctrl (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    output reg  refresh_trigger,",
        "    output reg  [31:0] refresh_count",
        ");",
        "",
        f"    localparam REFRESH_CYCLES = {cycles};",
        "    reg [31:0] counter;",
        "",
        "    always @(posedge clk or posedge rst) begin",
        "        if (rst) begin",
        "            counter <= 0;",
        "            refresh_trigger <= 0;",
        "            refresh_count <= 0;",
        "        end else begin",
        "            if (counter >= REFRESH_CYCLES) begin",
        "                counter <= 0;",
        "                refresh_trigger <= 1;",
        "                refresh_count <= refresh_count + 1;",
        "            end else begin",
        "                counter <= counter + 1;",
        "                refresh_trigger <= 0;",
        "            end",
        "        end",
        "    end",
        "",
        "endmodule",
    ]

    return DriftCompensator(
        refresh_interval_ms=round(refresh_ms, 2),
        drift_rate_per_day=drift_rate_per_day,
        compensation_method=compensation_method,
        verilog_controller="\n".join(v),
    )
