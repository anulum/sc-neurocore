// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for astrocyte

package services

import (
	"math"
	"testing"
)

// TestAstrocyteInputDrivesIp3AsThePythonReference checks the Python reference's state after ten
// steps at an input of 2.5.
func TestAstrocyteInputDrivesIp3AsThePythonReference(t *testing.T) {
	s := NewAstrocyteModel()
	for i := 0; i < 10; i++ {
		if _, err := s.Step(2.5); err != nil {
			t.Fatalf("step %d failed: %v", i, err)
		}
	}
	if math.Abs(s.Ip3-0.741475) > 5e-7 || math.Abs(s.Ca-0.075363) > 5e-7 {
		t.Fatalf("after 10 steps ca %.9f ip3 %.9f", s.Ca, s.Ip3)
	}
}

// TestAstrocyteRefusesInvalidInputAndKeepsTheState checks that a negative or non-finite input is
// refused and changes nothing.
func TestAstrocyteRefusesInvalidInputAndKeepsTheState(t *testing.T) {
	for _, current := range []float64{-0.1, math.NaN(), math.Inf(1)} {
		s := NewAstrocyteModel()
		ca, h, ip3 := s.Ca, s.H, s.Ip3
		if _, err := s.Step(current); err == nil {
			t.Fatalf("current %v was accepted", current)
		}
		if s.Ca != ca || s.H != h || s.Ip3 != ip3 {
			t.Fatalf("current %v changed the state", current)
		}
	}
}

// TestAstrocyteRefusesACalciumCandidateBeyondThePool checks that a step whose calcium would exceed
// C0 is refused and changes nothing.
func TestAstrocyteRefusesACalciumCandidateBeyondThePool(t *testing.T) {
	s := NewAstrocyteModel()
	s.H, s.Ip3, s.VEr = 1.0, 50.0, 1.0e9
	ca, h, ip3 := s.Ca, s.H, s.Ip3
	if _, err := s.Step(0.0); err == nil {
		t.Fatal("a calcium candidate beyond the pool was accepted")
	}
	if s.Ca != ca || s.H != h || s.Ip3 != ip3 {
		t.Fatal("a refused step changed the state")
	}
}

// TestAstrocyteRefusesInvalidParameters checks that a negative leak and a zero timestep are refused.
func TestAstrocyteRefusesInvalidParameters(t *testing.T) {
	s := NewAstrocyteModel()
	s.Leak = -1.0
	if _, err := s.Step(0.0); err == nil {
		t.Fatal("a negative leak was accepted")
	}
	s = NewAstrocyteModel()
	s.Dt = 0.0
	if _, err := s.Step(0.0); err == nil {
		t.Fatal("a zero timestep was accepted")
	}
}
