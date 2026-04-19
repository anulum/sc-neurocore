# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Software safety monitor (ported from neuro_safe_monitor.sv)

"""Software-in-the-loop simulation of the hardware safety monitor.

Mirrors the 6 formally proven properties from safety_bounds.lean and
the SystemVerilog neuro_safe_monitor module, enabling pre-silicon
validation of safety invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SafetyLimits:
    """Configurable safety thresholds matching SV parameters."""

    max_current: int = 0x7FFF
    max_voltage: int = 0xC000
    coherence_limit: int = 0x0100
    sc_denom: int = 0x0100
    lif_v_max: int = 0xC000


@dataclass
class SafetyMonitor:
    """Software mirror of the hardware neuro_safe_monitor.

    Enforces all 6 formally proven properties:
      [P1] monitor_soundness — halt when current/voltage/coherence out of bounds
      [P2] safe_transition — coherence must not decrease (monotone)
      [P3] sc_precision_bound — popcount must be in [0, N]
      [P4] sc_add_preserves_range — SC addition result ≤ denominator
      [P5] lif_membrane_bounded — membrane ≤ v_max
      [P6] correlation_range — |SCC numerator| ≤ denominator
    """

    limits: SafetyLimits = field(default_factory=SafetyLimits)
    halted: bool = False
    violation_flags: int = 0  # 6-bit sticky flags
    _prev_coherence: int = 0

    def reset(self) -> None:
        """Reset monitor state (equivalent to rst_n pulse)."""
        self.halted = False
        self.violation_flags = 0
        self._prev_coherence = 0

    def check(
        self,
        current: int = 0,
        voltage: int = 0,
        coherence: int = 0xFFFF,
        popcount_k: int = 0,
        sc_add_result: int = 0,
        membrane: int = 0,
        scc_numerator: int = 0,
        scc_denominator: int = 0x0100,
    ) -> bool:
        """Check all 6 safety properties. Returns True if any violation detected.

        Violation flags are sticky — once set, only ``reset()`` clears them.
        """
        violations = 0

        # [P1] monitor_soundness
        if current > self.limits.max_current or voltage > self.limits.max_voltage:
            violations |= 0b000001
        if coherence < self.limits.coherence_limit:
            violations |= 0b000001

        # [P2] safe_transition (monotone coherence)
        if coherence < self._prev_coherence:
            violations |= 0b000010
        self._prev_coherence = coherence

        # [P3] sc_precision_bound
        if popcount_k > self.limits.sc_denom:
            violations |= 0b000100

        # [P4] sc_add_preserves_range
        if sc_add_result > self.limits.sc_denom:
            violations |= 0b001000

        # [P5] lif_membrane_bounded
        if membrane > self.limits.lif_v_max:
            violations |= 0b010000

        # [P6] correlation_range
        if abs(scc_numerator) > scc_denominator:
            violations |= 0b100000

        self.violation_flags |= violations
        if violations:
            self.halted = True

        return violations > 0

    def property_names(self) -> list[str]:
        """Return names of violated properties."""
        names = []
        if self.violation_flags & 0b000001:
            names.append("P1:monitor_soundness")
        if self.violation_flags & 0b000010:
            names.append("P2:safe_transition")
        if self.violation_flags & 0b000100:
            names.append("P3:sc_precision_bound")
        if self.violation_flags & 0b001000:
            names.append("P4:sc_add_preserves_range")
        if self.violation_flags & 0b010000:
            names.append("P5:lif_membrane_bounded")
        if self.violation_flags & 0b100000:
            names.append("P6:correlation_range")
        return names
