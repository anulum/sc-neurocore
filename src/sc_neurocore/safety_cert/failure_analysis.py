# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""FMEDA records, aggregate failure analysis, and reliability metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from sc_neurocore.safety_cert.standards import SILLevel


class FailureCategory(Enum):
    """Failure-effect categories used by the FMEDA arithmetic."""

    SAFE = "safe"
    DANGEROUS_DETECTED = "dangerous_detected"
    DANGEROUS_UNDETECTED = "dangerous_undetected"
    NO_EFFECT = "no_effect"


@dataclass
class FailureMode:
    """One failure mode in the FMEDA."""

    fm_id: str
    component: str
    description: str
    category: FailureCategory
    failure_rate_fit: float  # FIT = failures per 10^9 hours
    diagnostic_coverage: float = 0.0  # 0.0 – 1.0
    mitigation: str = ""

    def __post_init__(self) -> None:
        """Validate one caller-supplied failure-mode record."""
        if not isinstance(self.fm_id, str) or not self.fm_id.strip():
            raise ValueError("fm_id must be a non-empty string")
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("component must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.category, FailureCategory):
            raise ValueError("category must be a FailureCategory")
        if isinstance(self.failure_rate_fit, bool) or not isinstance(
            self.failure_rate_fit, int | float
        ):
            raise ValueError("failure_rate_fit must be a finite non-negative number")
        if not math.isfinite(float(self.failure_rate_fit)) or float(self.failure_rate_fit) < 0.0:
            raise ValueError("failure_rate_fit must be a finite non-negative number")
        if isinstance(self.diagnostic_coverage, bool) or not isinstance(
            self.diagnostic_coverage, int | float
        ):
            raise ValueError("diagnostic_coverage must be a finite value in [0, 1]")
        if (
            not math.isfinite(float(self.diagnostic_coverage))
            or float(self.diagnostic_coverage) < 0.0
            or float(self.diagnostic_coverage) > 1.0
        ):
            raise ValueError("diagnostic_coverage must be a finite value in [0, 1]")
        if not isinstance(self.mitigation, str):
            raise ValueError("mitigation must be a string")

    @property
    def safe_failure_fraction(self) -> float:
        """Fraction of failures that are safe or detected dangerous."""
        if self.category in (FailureCategory.SAFE, FailureCategory.NO_EFFECT):
            return 1.0
        if self.category == FailureCategory.DANGEROUS_DETECTED:
            return self.diagnostic_coverage
        return 0.0


class FMEDA:
    """Failure Modes, Effects, and Diagnostic Analysis.

    Aggregates failure modes for SC neuromorphic modules and computes
    Safe Failure Fraction (SFF) and Diagnostic Coverage (DC).
    """

    def __init__(self) -> None:
        self.failure_modes: List[FailureMode] = []

    def add_failure_mode(self, fm: FailureMode) -> None:
        """Add one uniquely identified failure mode."""
        if not isinstance(fm, FailureMode):
            raise ValueError("fm must be a FailureMode")
        if any(existing.fm_id == fm.fm_id for existing in self.failure_modes):
            raise ValueError(f"failure mode already exists: {fm.fm_id}")
        self.failure_modes.append(fm)

    def add_sc_standard_modes(
        self,
        component: str,
        *,
        acknowledge_synthetic_profile: bool = False,
    ) -> None:
        """Add the legacy synthetic profile after explicit acknowledgement.

        The rates are test/demo assumptions without device, process, environment,
        or field-data provenance. They must never enter a safety case as measured
        FIT or diagnostic-coverage evidence.
        """
        if not isinstance(component, str) or not component.strip():
            raise ValueError("component must be a non-empty string")
        if not isinstance(acknowledge_synthetic_profile, bool):
            raise ValueError("acknowledge_synthetic_profile must be a boolean")
        if not acknowledge_synthetic_profile:
            raise ValueError(
                "synthetic SC failure profile requires acknowledge_synthetic_profile=True"
            )
        component = component.strip()
        modes = [
            FailureMode(
                f"{component}_LFSR_STUCK",
                component,
                "LFSR generator stuck at fixed value",
                FailureCategory.DANGEROUS_DETECTED,
                10.0,
                0.99,
                "Example candidate: ECC diagnostic; verification evidence required",
            ),
            FailureMode(
                f"{component}_BIT_FLIP",
                component,
                "Single-event upset in bitstream register",
                FailureCategory.DANGEROUS_DETECTED,
                50.0,
                0.95,
                "Example candidate: correlation monitor; verification evidence required",
            ),
            FailureMode(
                f"{component}_CLOCK_DRIFT",
                component,
                "Clock frequency deviation exceeds tolerance",
                FailureCategory.DANGEROUS_DETECTED,
                5.0,
                0.90,
                "Example candidate: watchdog; verification evidence required",
            ),
            FailureMode(
                f"{component}_WEIGHT_CORRUPT",
                component,
                "Q8.8 weight corruption in BRAM",
                FailureCategory.DANGEROUS_DETECTED,
                20.0,
                0.98,
                "Example candidate: range property; proof evidence required",
            ),
            FailureMode(
                f"{component}_SAFE_SILENT",
                component,
                "Neuron fails to spike (silent failure)",
                FailureCategory.SAFE,
                30.0,
                1.0,
                "Example candidate: firing-rate monitor; verification evidence required",
            ),
        ]
        for mode in modes:
            self.add_failure_mode(mode)

    @property
    def total_failure_rate(self) -> float:
        """Return the sum of caller-supplied FIT values."""
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
            if not math.isfinite(float(fm.failure_rate_fit)) or float(fm.failure_rate_fit) < 0.0:
                raise ValueError("failure_rate_fit must be a finite non-negative value")
        total = sum(fm.failure_rate_fit for fm in self.failure_modes)
        if not math.isfinite(float(total)) or total < 0.0:
            raise ValueError("total_failure_rate must be a finite non-negative value")
        return total

    @property
    def safe_failure_fraction(self) -> float:
        """SFF = (safe + no_effect + DC*dangerous_detected) / total."""
        if not self.failure_modes:
            return 0.0
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        total = self.total_failure_rate
        if total == 0:
            return 0.0
        safe_sum = sum(fm.failure_rate_fit * fm.safe_failure_fraction for fm in self.failure_modes)
        return safe_sum / total

    @property
    def diagnostic_coverage(self) -> float:
        """Return the FIT-weighted coverage of detected-dangerous modes."""
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        dd = [fm for fm in self.failure_modes if fm.category == FailureCategory.DANGEROUS_DETECTED]
        if not dd:
            return 0.0
        for fm in dd:
            if not math.isfinite(float(fm.diagnostic_coverage)):
                raise ValueError("diagnostic_coverage entries must be finite values")
            if float(fm.diagnostic_coverage) < 0.0 or float(fm.diagnostic_coverage) > 1.0:
                raise ValueError("diagnostic_coverage entries must be in [0, 1]")
        weighted_sum = sum(fm.diagnostic_coverage * fm.failure_rate_fit for fm in dd)
        denominator = sum(fm.failure_rate_fit for fm in dd)
        if not math.isfinite(float(weighted_sum)) or weighted_sum < 0.0:
            raise ValueError("diagnostic_coverage weighted sum must be a finite non-negative value")
        if not math.isfinite(float(denominator)) or denominator <= 0.0:
            raise ValueError("diagnostic_coverage denominator must be a finite positive value")
        return weighted_sum / denominator

    @property
    def residual_risk_fit(self) -> float:
        """Dangerous-undetected failure rate (residual risk)."""
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        residual = sum(
            fm.failure_rate_fit * (1.0 - fm.safe_failure_fraction) for fm in self.failure_modes
        )
        if not math.isfinite(float(residual)) or residual < 0.0:
            raise ValueError("residual_risk_fit must be a finite non-negative value")
        return residual

    def sff_by_component(self) -> Dict[str, float]:
        """Per-component safe failure fraction."""
        components: Dict[str, List[FailureMode]] = {}
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
            if not isinstance(fm.component, str) or not fm.component.strip():
                raise ValueError("failure modes must have non-empty component names")
            components.setdefault(fm.component, []).append(fm)
        result = {}
        for comp, fms in components.items():
            total = sum(f.failure_rate_fit for f in fms)
            safe = sum(f.failure_rate_fit * f.safe_failure_fraction for f in fms)
            if not math.isfinite(float(total)) or total < 0.0:
                raise ValueError("component failure-rate totals must be finite non-negative values")
            if not math.isfinite(float(safe)) or safe < 0.0:
                raise ValueError("component safe-failure totals must be finite non-negative values")
            result[comp] = safe / total if total > 0 else 0.0
        return result

    def max_achievable_sil(self) -> SILLevel:
        """Return a legacy SFF/DC screening label, not a SIL determination."""
        sff = self.safe_failure_fraction
        dc = self.diagnostic_coverage
        for value, name in ((sff, "safe_failure_fraction"), (dc, "diagnostic_coverage")):
            if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")
        if sff >= 0.99 and dc >= 0.99:
            return SILLevel.SIL_4
        if sff >= 0.97 and dc >= 0.99:
            return SILLevel.SIL_3
        if sff >= 0.90 and dc >= 0.90:
            return SILLevel.SIL_2
        if sff >= 0.60:
            return SILLevel.SIL_1
        return SILLevel.SIL_1

    def generate_report(self) -> str:
        """Render the supplied FMEDA arithmetic and its provenance warning."""
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        if not self.failure_modes:
            return (
                "# FMEDA Report\n\n"
                "Status: not assessed. No caller-supplied failure modes or FIT data were provided."
            )
        lines = [
            "# FMEDA Report",
            "Input provenance: caller supplied; independent review required.",
            f"Total failure rate: {self.total_failure_rate:.1f} FIT",
            f"Safe Failure Fraction: {self.safe_failure_fraction:.1%}",
            f"Diagnostic Coverage: {self.diagnostic_coverage:.1%}",
            f"Max achievable SIL: SIL {self.max_achievable_sil().value}",
            "",
            "| FM ID | Component | Category | Rate (FIT) | DC | Mitigation |",
            "|-------|-----------|----------|------------|-----|------------|",
        ]
        for fm in sorted(self.failure_modes, key=lambda mode: mode.fm_id):
            lines.append(
                f"| {fm.fm_id} | {fm.component} | {fm.category.value} "
                f"| {fm.failure_rate_fit:.1f} | {fm.diagnostic_coverage:.0%} | {fm.mitigation} |"
            )
        return "\n".join(lines)


@dataclass
class ReliabilityMetrics:
    """System-level reliability from FMEDA data."""

    total_fit: float
    dangerous_undetected_fit: float

    def __post_init__(self) -> None:
        """Validate system-level FIT values."""
        if isinstance(self.total_fit, bool) or not isinstance(self.total_fit, int | float):
            raise ValueError("total_fit must be a finite non-negative number")
        if not math.isfinite(float(self.total_fit)) or float(self.total_fit) < 0.0:
            raise ValueError("total_fit must be a finite non-negative number")
        if isinstance(self.dangerous_undetected_fit, bool) or not isinstance(
            self.dangerous_undetected_fit, int | float
        ):
            raise ValueError("dangerous_undetected_fit must be a finite non-negative number")
        if (
            not math.isfinite(float(self.dangerous_undetected_fit))
            or float(self.dangerous_undetected_fit) < 0.0
        ):
            raise ValueError("dangerous_undetected_fit must be a finite non-negative number")
        if float(self.dangerous_undetected_fit) > float(self.total_fit):
            raise ValueError("dangerous_undetected_fit cannot exceed total_fit")

    @property
    def mtbf_hours(self) -> float:
        """Mean Time Between Failures (hours)."""
        if self.total_fit <= 0:
            return float("inf")
        return 1e9 / self.total_fit

    @property
    def mtbf_years(self) -> float:
        """Return mean time between failures in 365-day years."""
        return self.mtbf_hours / 8760.0

    @property
    def pfh_d(self) -> float:
        """Probability of dangerous failure per hour."""
        if self.dangerous_undetected_fit <= 0:
            return 0.0
        return self.dangerous_undetected_fit / 1e9

    @property
    def pfh_sil(self) -> SILLevel:
        """Return the legacy PFH-only screening label, not a SIL determination."""
        pfh = self.pfh_d
        if not math.isfinite(float(pfh)) or float(pfh) < 0.0:
            raise ValueError("pfh_d must be a finite non-negative number")
        if pfh <= 1e-8:
            return SILLevel.SIL_4
        if pfh <= 1e-7:
            return SILLevel.SIL_3
        if pfh <= 1e-6:
            return SILLevel.SIL_2
        return SILLevel.SIL_1

    @staticmethod
    def from_fmeda(fmeda: FMEDA) -> ReliabilityMetrics:
        """Build metrics from explicitly populated FMEDA arithmetic."""
        if not isinstance(fmeda, FMEDA):
            raise ValueError("fmeda must be an FMEDA")
        return ReliabilityMetrics(
            total_fit=fmeda.total_failure_rate,
            dangerous_undetected_fit=fmeda.residual_risk_fit,
        )


__all__ = [
    "FailureCategory",
    "FailureMode",
    "FMEDA",
    "ReliabilityMetrics",
]
