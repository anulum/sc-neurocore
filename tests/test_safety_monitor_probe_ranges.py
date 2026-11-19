# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The software safety monitor takes only values its hardware probes can carry

"""Hold the software safety monitor to the probe ranges of ``safety_monitor.sv``.

The hardware probes are 16-bit wires: unsigned, except the signed SCC numerator. The software mirror once took any
number, so a negative current, voltage, popcount, SC sum or membrane passed every upper-bound check and reported no
violation. These tests hold the refusal of every value outside its probe's range before any state changes, the
accepted range ends, the RTL's own handling of a zero SCC denominator, and the docstring's account of which
properties ``safety_bounds.lean`` proves and which it takes as axioms.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.safety_cert.safety_monitor import SafetyMonitor, _refuse_out_of_range

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "src" / "sc_neurocore" / "formal" / "proofs" / "safety_bounds.lean"
RTL = ROOT / "src" / "sc_neurocore" / "hdl_gen" / "safety" / "safety_monitor.sv"

UNSIGNED = [
    "current",
    "voltage",
    "coherence",
    "popcount_k",
    "sc_add_result",
    "membrane",
    "scc_denominator",
]


def _untouched(mon: SafetyMonitor) -> bool:
    """Return whether the monitor still holds its reset state."""
    return not mon.halted and mon.violation_flags == 0 and mon._prev_coherence == 0


@pytest.mark.parametrize("probe", UNSIGNED)
@pytest.mark.parametrize("value", [-1, -999_999, 0x10000])
def test_an_unsigned_probe_outside_16_bits_is_refused_before_any_state_changes(
    probe: str, value: int
) -> None:
    """A negative or 17-bit value on an unsigned probe raises, names the probe and its range, and changes nothing."""
    mon = SafetyMonitor()
    with pytest.raises(ValueError, match=rf"{probe}={value} .*\[0, 65535\]"):
        mon.check(**{probe: value})
    assert _untouched(mon)


@pytest.mark.parametrize("value", [-0x8001, 0x8000, 999_999])
def test_the_signed_scc_numerator_outside_16_bits_is_refused(value: int) -> None:
    """The SCC numerator is a signed 16-bit probe: outside [-32768, 32767] it raises and nothing changes."""
    mon = SafetyMonitor()
    with pytest.raises(ValueError, match=r"scc_numerator=.*\[-32768, 32767\]"):
        mon.check(scc_numerator=value)
    assert _untouched(mon)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_value_that_is_not_a_number_or_infinite_is_refused(value: float) -> None:
    """NaN and infinity lie in no probe range."""
    mon = SafetyMonitor()
    with pytest.raises(ValueError, match="membrane="):
        mon.check(membrane=cast(int, value))
    assert _untouched(mon)


def test_the_range_ends_are_taken_and_judged_as_in_the_rtl() -> None:
    """Zero and 0xFFFF are valid on every unsigned probe; -32768 and 32767 on the numerator."""
    mon = SafetyMonitor()
    zeros = dict.fromkeys(UNSIGNED, 0)
    zeros["coherence"] = 0xFFFF
    zeros["scc_denominator"] = 0xFFFF
    assert mon.check(**zeros, scc_numerator=-0x8000) is False
    assert mon.check(**dict.fromkeys(UNSIGNED, 0xFFFF), scc_numerator=0x7FFF) is True
    assert mon.property_names() == [
        "P1:monitor_soundness",
        "P3:sc_precision_bound",
        "P4:sc_add_preserves_range",
        "P5:lif_membrane_bounded",
    ]


def test_a_zero_scc_denominator_violates_only_with_a_nonzero_numerator() -> None:
    """As in the RTL (``abs_scc_numer > probe_scc_denom``), 0/0 passes and any non-zero numerator over 0 violates."""
    assert "abs_scc_numer > probe_scc_denom" in RTL.read_text(encoding="utf-8")
    mon = SafetyMonitor()
    assert mon.check(scc_numerator=0, scc_denominator=0) is False
    assert mon.check(scc_numerator=-1, scc_denominator=0) is True
    assert mon.property_names() == ["P6:correlation_range"]


def test_the_docstrings_state_which_properties_are_proved_and_which_are_axioms() -> None:
    """Four of the six properties are theorems of ``safety_bounds.lean`` and two are axioms; the docstrings say so."""
    lean = LEAN.read_text(encoding="utf-8")
    axioms = set(re.findall(r"^axiom\s+(\w+)", lean, flags=re.MULTILINE))
    assert axioms == {"sc_precision_numerator_bound", "sc_add_preserves_range"}
    for theorem in ("monitor_soundness", "safe_transition", "lif_membrane_bounded", "scc_bounded"):
        assert re.search(rf"^theorem\s+{theorem}\b", lean, flags=re.MULTILINE), theorem
    import sc_neurocore.safety_cert.safety_monitor as module

    for text in (module.__doc__ or "", SafetyMonitor.__doc__ or ""):
        assert "6 formally proven" not in text
    doc = SafetyMonitor.__doc__ or ""
    assert "four are proved" in doc and "two are axioms" in doc
    for name in axioms:
        assert name in doc


def test_the_refusal_names_the_first_probe_out_of_range_and_passes_a_valid_set() -> None:
    """The refusal reports probes in their argument order and stays silent when every value is within range."""
    with pytest.raises(ValueError, match=r"^voltage=-1 "):
        _refuse_out_of_range(current=0, voltage=-1, membrane=-1)
    _refuse_out_of_range(current=0, scc_numerator=-0x8000, scc_denominator=0xFFFF)


def test_integrality_is_not_checked_and_a_fraction_in_range_is_compared_as_a_number() -> None:
    """The probes are annotated ``int``; a fraction inside a range is not refused but compared as it stands."""
    mon = SafetyMonitor()
    assert mon.check(current=cast(int, 0x7FFF + 0.5)) is True
    assert mon.property_names() == ["P1:monitor_soundness"]
    mon.reset()
    assert mon.check(current=cast(int, 300.7)) is False
