# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for software safety monitor

"""Tests for software safety monitor — mirrors SV testbench."""

from sc_neurocore.safety_cert.safety_monitor import SafetyMonitor


class TestSafetyMonitor:
    def setup_method(self):
        self.mon = SafetyMonitor()

    def test_normal_operation(self):
        violated = self.mon.check(
            current=0x1000,
            voltage=0x2000,
            coherence=0xFFFF,
            popcount_k=0x0080,
            sc_add_result=0x0080,
            membrane=0x4000,
            scc_numerator=0x50,
            scc_denominator=0x100,
        )
        assert not violated
        assert not self.mon.halted

    def test_p1_current_overflow(self):
        self.mon.check(current=0xFFFF, coherence=0xFFFF)
        assert self.mon.halted
        assert "P1:monitor_soundness" in self.mon.property_names()

    def test_p1_voltage_overflow(self):
        self.mon.check(voltage=0xFFFF, coherence=0xFFFF)
        assert self.mon.halted

    def test_p1_coherence_violation(self):
        self.mon.check(coherence=0x0010)
        assert self.mon.halted
        assert self.mon.violation_flags & 0b000001

    def test_p2_monotone_coherence(self):
        self.mon.check(coherence=0xF000)
        self.mon.check(coherence=0x0F00)  # decreased
        assert self.mon.halted
        assert "P2:safe_transition" in self.mon.property_names()

    def test_p3_precision_violation(self):
        self.mon.check(popcount_k=0x0200, coherence=0xFFFF)
        assert self.mon.halted
        assert "P3:sc_precision_bound" in self.mon.property_names()

    def test_p4_sc_range_violation(self):
        self.mon.check(sc_add_result=0x0200, coherence=0xFFFF)
        assert self.mon.halted
        assert "P4:sc_add_preserves_range" in self.mon.property_names()

    def test_p5_membrane_violation(self):
        self.mon.check(membrane=0xFFFF, coherence=0xFFFF)
        assert self.mon.halted
        assert "P5:lif_membrane_bounded" in self.mon.property_names()

    def test_p6_scc_positive_overflow(self):
        self.mon.check(scc_numerator=0x200, scc_denominator=0x100, coherence=0xFFFF)
        assert self.mon.halted
        assert "P6:correlation_range" in self.mon.property_names()

    def test_p6_scc_negative_overflow(self):
        self.mon.check(scc_numerator=-512, scc_denominator=256, coherence=0xFFFF)
        assert self.mon.halted

    def test_exact_boundaries_no_violation(self):
        self.mon.check(
            current=0x7FFF,
            voltage=0xC000,
            coherence=0x0100,
            popcount_k=0x0100,
            sc_add_result=0x0100,
            membrane=0xC000,
            scc_numerator=0x100,
            scc_denominator=0x100,
        )
        assert not self.mon.halted

    def test_sticky_flags(self):
        self.mon.check(current=0xFFFF, coherence=0xFFFF)
        assert self.mon.violation_flags & 0b000001
        self.mon.check(current=0x0000, coherence=0xFFFF)
        assert self.mon.violation_flags & 0b000001  # still sticky

    def test_reset_clears_all(self):
        self.mon.check(current=0xFFFF, coherence=0xFFFF)
        assert self.mon.halted
        self.mon.reset()
        assert not self.mon.halted
        assert self.mon.violation_flags == 0
