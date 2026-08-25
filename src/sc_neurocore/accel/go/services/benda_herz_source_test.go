// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import "testing"

func TestBendaHerzSourceDefaults(t *testing.T) {
	s := NewBendaHerzNeuron()
	if !s.Valid() || s.OnsetGain != 60.0 || s.AdaptationSlope != 0.1 {
		t.Fatal("invalid source defaults")
	}
}

func TestBendaHerzSourcePhaseReset(t *testing.T) {
	s := NewBendaHerzNeuron()
	s.Phase = 0.99
	s.Dt = 1.0
	s.AdaptationSlope = 0.0
	if got := s.Step(1.0); got != 1 || s.Phase != 0.0 {
		t.Fatalf("got spike=%d phase=%g", got, s.Phase)
	}
}
