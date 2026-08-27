// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Larter-Breakspear dual-identity parity anchors

package services

import (
	"math"
	"testing"
)

func TestLarterBreakspearSourceAnchor(t *testing.T) {
	state := NewLarterBreakspearNeuron()
	value, err := state.TryStep(0.0)
	if err != nil || math.Abs(value-0.10851023903311285) > 2e-15 {
		t.Fatalf("source anchor mismatch: value=%g err=%v", value, err)
	}
	before := *state
	if _, err = state.TryStep(math.NaN()); err == nil || *state != before {
		t.Fatal("invalid source input did not fail atomically")
	}
}

func TestSCDecoupledAdaptationIonMassAnchor(t *testing.T) {
	state := NewSCDecoupledAdaptationIonMassNeuron()
	value, err := state.TryStep(0.0)
	if err != nil || math.Abs(value-(-0.4987593419078305)) > 2e-15 {
		t.Fatalf("SC anchor mismatch: value=%g err=%v", value, err)
	}
}

func BenchmarkLarterBreakspearRK4(b *testing.B) {
	state := NewLarterBreakspearNeuron()
	b.ResetTimer()
	for range b.N {
		if _, err := state.TryStep(0.0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSCDecoupledAdaptationIonMassRK4(b *testing.B) {
	state := NewSCDecoupledAdaptationIonMassNeuron()
	b.ResetTimer()
	for range b.N {
		if _, err := state.TryStep(0.0); err != nil {
			b.Fatal(err)
		}
	}
}
