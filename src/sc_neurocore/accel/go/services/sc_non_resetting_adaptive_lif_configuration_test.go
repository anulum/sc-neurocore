// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained adaptive LIF service configuration contracts

package services

import (
	"math"
	"testing"
)

// TestSCNonResettingAdaptiveLIFAllFields verifies refusal after every nonfinite edit.
func TestSCNonResettingAdaptiveLIFAllFields(t *testing.T) {
	for index := 0; index < 9; index++ {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			s := NewSCNonResettingAdaptiveLIFNeuron()
			fields := []*float64{&s.V, &s.Theta, &s.VRest, &s.ThetaRest, &s.DeltaTheta, &s.TauM, &s.TauTheta, &s.RM, &s.Dt}
			*fields[index] = bad
			v, theta := math.Float64bits(s.V), math.Float64bits(s.Theta)
			if s.Valid() {
				t.Fatalf("field %d accepted nonfinite configuration", index)
			}
			if _, err := s.Step(20); err != ErrSCNonResettingAdaptiveLIFInvalidState {
				t.Fatalf("field %d refusal: %v", index, err)
			}
			if math.Float64bits(s.V) != v || math.Float64bits(s.Theta) != theta {
				t.Fatalf("field %d partially committed", index)
			}
		}
	}
}

// TestSCNonResettingAdaptiveLIFReset verifies atomic refusal and dynamic-state recovery.
func TestSCNonResettingAdaptiveLIFReset(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		s := NewSCNonResettingAdaptiveLIFNeuron()
		s.V, s.Theta, s.VRest = -64, -49, bad
		if err := s.TryReset(); err != ErrSCNonResettingAdaptiveLIFInvalidState {
			t.Fatalf("invalid reset accepted: %v", err)
		}
		s.Reset()
		if s.V != -64 || s.Theta != -49 {
			t.Fatal("invalid reset changed a dynamic field")
		}
		s.VRest, s.V, s.Theta = -65, math.NaN(), math.Inf(1)
		if err := s.TryReset(); err != nil || s.V != -65 || s.Theta != -50 {
			t.Fatalf("valid reset failed recovery: %v", err)
		}
		if _, err := s.Step(20); err != nil {
			t.Fatal(err)
		}
	}
}

// TestSCNonResettingAdaptiveLIFFiniteOverflow verifies candidate refusal and retry.
func TestSCNonResettingAdaptiveLIFFiniteOverflow(t *testing.T) {
	s := NewSCNonResettingAdaptiveLIFNeuron()
	s.RM = 1e308
	if _, err := s.Step(20); err != ErrSCNonResettingAdaptiveLIFNonFiniteUpdate || s.V != -65 || s.Theta != -50 {
		t.Fatalf("steady-state overflow committed state: %v", err)
	}
	if event, err := s.Step(0); event != 0 || err != nil {
		t.Fatalf("zero-current retry failed: %v", err)
	}
	s.V, s.Theta, s.ThetaRest, s.DeltaTheta = 1.5e308, 1e308, 1e308, 1e308
	if _, err := s.Step(0); err != ErrSCNonResettingAdaptiveLIFNonFiniteUpdate || s.V != 1.5e308 || s.Theta != 1e308 {
		t.Fatalf("threshold overflow committed state: %v", err)
	}
}
