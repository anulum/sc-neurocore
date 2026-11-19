# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Software safety monitor (ported from neuro_safe_monitor.sv)

"""Software-in-the-loop simulation of the hardware safety monitor.

Mirrors the run-time checks of the SystemVerilog ``safety_monitor.sv`` for the six properties named after
``safety_bounds.lean`` (four proved there, two stated as axioms), enabling pre-silicon validation of the safety
invariants. Inputs are taken only within the ranges the hardware probes can carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Range of the unsigned 16-bit probes of ``safety_monitor.sv``.
UNSIGNED_PROBE_RANGE = (0, 0xFFFF)
#: Range of the signed 16-bit SCC numerator probe.
SIGNED_PROBE_RANGE = (-0x8000, 0x7FFF)


def _refuse_out_of_range(**probes: int) -> None:
    """Raise ``ValueError`` naming the first probe whose value lies outside its 16-bit range."""
    for name, value in probes.items():
        lo, hi = SIGNED_PROBE_RANGE if name == "scc_numerator" else UNSIGNED_PROBE_RANGE
        if not lo <= value <= hi:
            raise ValueError(
                f"{name}={value} lies outside its 16-bit hardware probe range [{lo}, {hi}]"
            )


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
    """Software mirror of the hardware safety monitor ``safety_monitor.sv``.

    Checks at run time the six properties named after ``safety_bounds.lean``. Of these, four are proved there
    (``monitor_soundness``, ``safe_transition``, ``lif_membrane_bounded``, ``scc_bounded``) and two are axioms
    pending Mathlib's non-linear arithmetic (``sc_precision_numerator_bound``, ``sc_add_preserves_range``):

      [P1] monitor_soundness — halt when current or voltage exceeds its limit or coherence is below its limit
      [P2] safe_transition — coherence must not decrease (monotone)
      [P3] sc_precision_bound — popcount ≤ N, the hypothesis of the axiomatised precision bound
      [P4] sc_add_preserves_range — SC addition result ≤ denominator
      [P5] lif_membrane_bounded — membrane ≤ v_max
      [P6] correlation_range — |SCC numerator| ≤ denominator

    Every probe is a 16-bit value as in the hardware: unsigned, except the signed SCC numerator. A value outside
    its probe's range, NaN and infinity included, raises ``ValueError`` before any state changes. Integrality
    is not checked: the probes are annotated ``int``, and a fraction inside a range is compared as it stands. As in the RTL, a zero SCC denominator is a
    violation only with a non-zero numerator (``scc_bounded`` itself assumes a positive denominator).
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

        Raises
        ------
        ValueError
            If a value lies outside its 16-bit probe range (NaN and infinity included); nothing changes then.
        """
        lo, hi = UNSIGNED_PROBE_RANGE
        if not (
            lo <= current <= hi
            and lo <= voltage <= hi
            and lo <= coherence <= hi
            and lo <= popcount_k <= hi
            and lo <= sc_add_result <= hi
            and lo <= membrane <= hi
            and lo <= scc_denominator <= hi
            and SIGNED_PROBE_RANGE[0] <= scc_numerator <= SIGNED_PROBE_RANGE[1]
        ):
            _refuse_out_of_range(
                current=current,
                voltage=voltage,
                coherence=coherence,
                popcount_k=popcount_k,
                sc_add_result=sc_add_result,
                membrane=membrane,
                scc_numerator=scc_numerator,
                scc_denominator=scc_denominator,
            )
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
