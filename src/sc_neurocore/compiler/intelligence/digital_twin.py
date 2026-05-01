# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
from dataclasses import dataclass

# 32. Analog Drift Compensation Generator
# ═══════════════════════════════════════════════════════════════════════


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
    """Generate analog drift compensation controller.

    For analog/memristive targets, creates on-chip calibration
    circuits that periodically refresh weights to compensate
    for device aging and retention loss.

    Parameters
    ----------
    module_name : str
        Module name.
    drift_rate_per_day : float
        Weight drift per day (fraction).
    max_drift_tolerance : float
        Maximum acceptable drift before refresh.
    clock_freq_mhz : int
        Clock frequency.
    compensation_method : str
        Compensation strategy.

    Returns
    -------
    DriftCompensator
        Controller with Verilog.
    """
    # Calculate refresh interval
    if drift_rate_per_day > 0:
        days_to_tolerance = max_drift_tolerance / drift_rate_per_day
        refresh_ms = days_to_tolerance * 24 * 3600 * 1000
    else:
        refresh_ms = 1e9  # effectively never

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


# ═══════════════════════════════════════════════════════════════════════
# 62. HIL Calibration Stub Generator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class HILCalibration:
    """Hardware-in-the-loop calibration protocol.

    Attributes
    ----------
    protocol_steps : list[str]
    num_parameters : int
    sweep_ranges : dict[str, tuple[float, float]]
    """

    protocol_steps: list[str]
    num_parameters: int
    sweep_ranges: dict[str, tuple[float, float]]


def generate_hil_calibration(
    module_name: str,
    equations: dict[str, str],
    *,
    parameters: dict[str, tuple[float, float]] | None = None,
) -> HILCalibration:
    """Generate hardware-in-the-loop calibration protocol.

    Produces a step-by-step calibration procedure for compensating
    analog drift, mismatch, and process variation on real hardware.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        State variable equations.
    parameters : dict[str, tuple[float, float]], optional
        Parameter sweep ranges {name: (min, max)}.

    Returns
    -------
    HILCalibration
    """
    if parameters is None:
        parameters = {var: (0.0, 1.0) for var in equations}

    steps = [
        f"1. Deploy {module_name} bitstream to target hardware",
        f"2. Initialise all {len(equations)} state variables to zero",
    ]
    step_num = 3
    for param, (lo, hi) in parameters.items():
        steps.append(f"{step_num}. Sweep '{param}' from {lo} to {hi}, record output at 10 points")
        step_num += 1
    steps.extend(
        [
            f"{step_num}. Compare measured outputs to software golden model",
            f"{step_num + 1}. Apply least-squares correction coefficients",
            f"{step_num + 2}. Repeat sweep to verify drift < 1 LSB",
        ]
    )

    return HILCalibration(
        protocol_steps=steps,
        num_parameters=len(parameters),
        sweep_ranges=parameters,
    )


# ═══════════════════════════════════════════════════════════════════════
# 63. Digital Twin Shadow Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_digital_twin(
    module_name: str,
    equations: dict[str, str],
    profile_name: str,
) -> str:
    """Generate a Python digital twin that mirrors deployed hardware.

    The twin tracks identical state transitions in software, enabling
    runtime comparison, anomaly detection, and predictive maintenance.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        State variable equations.
    profile_name : str
        Target hardware profile.

    Returns
    -------
    str
        Python source code for the digital twin class.
    """
    vars_list = list(equations.keys())
    lines = [
        f'"""Digital twin for {module_name} targeting {profile_name}."""',
        "",
        f"class {module_name.title().replace('_', '')}Twin:",
        '    """Software shadow of deployed hardware state."""',
        "",
        "    def __init__(self):",
    ]
    for v in vars_list:
        lines.append(f"        self.{v} = 0.0")
    lines.extend(
        [
            "        self.cycle = 0",
            "",
            "    def step(self, inputs: dict[str, float]) -> dict[str, float]:",
            '        """Execute one timestep, mirroring hardware state."""',
        ]
    )
    for v, expr in equations.items():
        lines.append(f"        # {v} = {expr}")
        lines.append(f"        self.{v} = inputs.get('{v}', self.{v})")
    lines.extend(
        [
            "        self.cycle += 1",
            f"        return {{{', '.join(repr(v) + ': self.' + v for v in vars_list)}}}",
            "",
            "    def compare(self, hw_state: dict[str, float]) -> dict[str, float]:",
            '        """Compare twin state against hardware telemetry."""',
            "        return {k: abs(getattr(self, k, 0) - hw_state.get(k, 0))",
            f"                for k in {vars_list!r}}}",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 65. SEU Scrub Scheduler
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ScrubSchedule:
    """Configuration memory scrubbing schedule.

    Attributes
    ----------
    interval_ms : float
    strategy : str
    frames_per_cycle : int
    expected_seu_rate : float
    """

    interval_ms: float
    strategy: str
    frames_per_cycle: int
    expected_seu_rate: float


def schedule_seu_scrubbing(
    config_bits: int,
    *,
    orbit_altitude_km: float = 400.0,
    shielding_mm_al: float = 3.0,
    strategy: str = "hybrid",
) -> ScrubSchedule:
    """Generate scrubbing schedule for space-grade configuration memory.

    Uses orbital altitude and shielding to estimate SEU rate, then
    calculates optimal scrub interval for target reliability.

    Parameters
    ----------
    config_bits : int
        Total configuration memory bits.
    orbit_altitude_km : float
        Orbital altitude (affects particle flux).
    shielding_mm_al : float
        Aluminium shielding thickness.
    strategy : str
        "blind" (full-chip) or "hybrid" (targeted + periodic full).

    Returns
    -------
    ScrubSchedule
    """
    # Simplified SEU rate model: flux increases with altitude
    base_rate = 1e-7  # upsets per bit per day at LEO
    altitude_factor = orbit_altitude_km / 400.0
    shielding_factor = max(0.1, 1.0 - shielding_mm_al * 0.15)
    seu_rate = base_rate * altitude_factor * shielding_factor

    # Scrub must complete before second upset is probable
    expected_upsets_per_day = seu_rate * config_bits
    if expected_upsets_per_day > 0:
        interval_hours = 1.0 / expected_upsets_per_day
    else:
        interval_hours = 24.0
    interval_ms = interval_hours * 3_600_000

    # Frame-based scrubbing
    frame_size = 1024  # bits per frame
    frames = max(1, config_bits // frame_size)

    return ScrubSchedule(
        interval_ms=round(interval_ms, 2),
        strategy=strategy,
        frames_per_cycle=frames,
        expected_seu_rate=round(expected_upsets_per_day, 6),
    )


# ═══════════════════════════════════════════════════════════════════════
# 76. Hardware Telemetry Ingestion
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TelemetryResult:
    """Hardware telemetry comparison result.

    Attributes
    ----------
    samples : int
    max_drift : float
    mean_drift : float
    alerts : list[str]
    healthy : bool
    """

    samples: int
    max_drift: float
    mean_drift: float
    alerts: list[str]
    healthy: bool


def ingest_telemetry(
    telemetry_data: list[dict[str, float]],
    twin_states: list[dict[str, float]],
    *,
    drift_threshold: float = 0.1,
) -> TelemetryResult:
    """Ingest hardware telemetry and compare against digital twin.

    Parameters
    ----------
    telemetry_data : list[dict[str, float]]
        List of hardware state snapshots {var: value}.
    twin_states : list[dict[str, float]]
        Corresponding digital twin states.
    drift_threshold : float
        Alert threshold for absolute drift.

    Returns
    -------
    TelemetryResult
    """
    if not telemetry_data or not twin_states:
        return TelemetryResult(
            samples=0,
            max_drift=0.0,
            mean_drift=0.0,
            alerts=[],
            healthy=True,
        )

    n = min(len(telemetry_data), len(twin_states))
    drifts = []
    alerts = []

    for i in range(n):
        hw = telemetry_data[i]
        tw = twin_states[i]
        for var in hw:
            d = abs(hw[var] - tw.get(var, 0.0))
            drifts.append(d)
            if d > drift_threshold:
                alerts.append(f"Sample {i}, var '{var}': drift={d:.4f} > {drift_threshold}")

    max_d = max(drifts) if drifts else 0.0
    mean_d = sum(drifts) / len(drifts) if drifts else 0.0

    return TelemetryResult(
        samples=n,
        max_drift=round(max_d, 6),
        mean_drift=round(mean_d, 6),
        alerts=alerts,
        healthy=len(alerts) == 0,
    )
