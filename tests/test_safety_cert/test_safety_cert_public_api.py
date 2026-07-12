# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.safety_cert package public API

"""Pin the safety package re-export and runtime-monitor contracts.

Every documented symbol must import from the package, retain identity with the
historical compatibility module, and remain listed in the public export list.
"""

from __future__ import annotations

import sc_neurocore.safety_cert as sc
from sc_neurocore.safety_cert import safety_cert as inner
from sc_neurocore.safety_cert import safety_monitor as monitor


SAFETY_CERT_SYMBOLS: tuple[str, ...] = (
    "ASILLevel",
    "CCFAnalysis",
    "CCFDefence",
    "CertificationGenerator",
    "CertificationPackage",
    "ChangeImpactTracker",
    "ChangeRecord",
    "ChecklistItem",
    "ComplianceChecklist",
    "CrossStandardMapper",
    "EvidenceBag",
    "EvidenceItem",
    "FMEDA",
    "FailureCategory",
    "FailureMode",
    "FormalProofCertificate",
    "FormalProperty",
    "FormalPropertyGapDetector",
    "HFTAssessment",
    "HFTLevel",
    "IEC62304Assessment",
    "ProofTestCoverage",
    "PropertyGap",
    "ReliabilityMetrics",
    "Requirement",
    "SafetyManualGenerator",
    "SafetyStandard",
    "SILLevel",
    "SWClass",
    "TraceabilityMatrix",
    "WCETAnalyzer",
    "WCETPath",
)

SAFETY_MONITOR_SYMBOLS: tuple[str, ...] = (
    "SafetyLimits",
    "SafetyMonitor",
)


# ───────────────────────── package-level metadata ─────────────────────────


def test_tier_is_industrial() -> None:
    """Package declares its tier as 'industrial' (regulatory work)."""
    assert sc.__tier__ == "industrial"


def test_all_is_a_concrete_list() -> None:
    """Package `__all__` is a list (not a generator/tuple comprehension)."""
    assert isinstance(sc.__all__, list)
    assert len(sc.__all__) == 34


# ───────────────────────── safety_cert.py re-exports ─────────────────────────


def test_safety_cert_symbol_count() -> None:
    """All 32 safety_cert.py public symbols re-exported."""
    for sym in SAFETY_CERT_SYMBOLS:
        assert hasattr(sc, sym), f"package missing {sym!r}"


def test_safety_cert_symbol_identity() -> None:
    """Top-level symbol IS the inner-module symbol (no shadow / no rebind)."""
    for sym in SAFETY_CERT_SYMBOLS:
        assert getattr(sc, sym) is getattr(inner, sym), (
            f"{sym!r} differs between package and inner module"
        )


def test_safety_cert_symbols_in_all() -> None:
    """Every safety_cert.py symbol appears in package __all__."""
    public = set(sc.__all__)
    missing = [s for s in SAFETY_CERT_SYMBOLS if s not in public]
    assert not missing, f"safety_cert symbols missing from __all__: {missing}"


# ───────────────────────── safety_monitor.py re-exports ─────────────────────────


def test_safety_monitor_symbol_count() -> None:
    """Both safety_monitor.py public symbols re-exported."""
    for sym in SAFETY_MONITOR_SYMBOLS:
        assert hasattr(sc, sym), f"package missing {sym!r}"


def test_safety_monitor_symbol_identity() -> None:
    """Top-level symbol IS the inner-module symbol."""
    for sym in SAFETY_MONITOR_SYMBOLS:
        assert getattr(sc, sym) is getattr(monitor, sym), (
            f"{sym!r} differs between package and safety_monitor module"
        )


def test_safety_monitor_symbols_in_all() -> None:
    public = set(sc.__all__)
    missing = [s for s in SAFETY_MONITOR_SYMBOLS if s not in public]
    assert not missing, f"safety_monitor symbols missing from __all__: {missing}"


# ───────────────────────── instantiability smoke ─────────────────────────


def test_safety_monitor_instantiates_with_defaults() -> None:
    """SafetyMonitor() with default limits is a valid construction."""
    mon = sc.SafetyMonitor()
    assert mon.halted is False
    assert mon.violation_flags == 0
    assert isinstance(mon.limits, sc.SafetyLimits)


def test_safety_monitor_check_returns_bool() -> None:
    """`check()` with default-safe inputs returns False (no violations)."""
    mon = sc.SafetyMonitor()
    # All-defaults call: zeros for current/voltage/popcount/sc_add/membrane,
    # max coherence (0xFFFF), zero numerator. Nothing should violate.
    result = mon.check()
    assert isinstance(result, bool)
    assert result is False
    assert mon.halted is False


def test_safety_monitor_check_flags_overcurrent() -> None:
    """Current above max_current must flip [P1]."""
    mon = sc.SafetyMonitor()
    result = mon.check(current=0x8000)  # > default max 0x7FFF
    assert result is True
    assert mon.halted is True
    assert mon.violation_flags & 0b000001
    assert "P1:monitor_soundness" in mon.property_names()


def test_safety_monitor_violation_flags_sticky_until_reset() -> None:
    """Once violated, flags persist; reset() clears them."""
    mon = sc.SafetyMonitor()
    mon.check(current=0x8000)
    assert mon.halted is True
    mon.reset()
    assert mon.halted is False
    assert mon.violation_flags == 0


def test_safety_standard_enum_has_documented_members() -> None:
    """SafetyStandard enum has the IEC/ISO/FDA members the docstring claims."""
    members = {m.name for m in sc.SafetyStandard}
    # Must at least cover the standards listed in the package docstring.
    assert "IEC_61508" in members or any("61508" in m for m in members)
    assert "ISO_26262" in members or any("26262" in m for m in members)


def test_sil_level_enum_4_levels() -> None:
    """SIL grading is SIL 1 / 2 / 3 / 4 (per IEC 61508-1)."""
    assert len(list(sc.SILLevel)) == 4
