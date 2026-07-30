// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go source MAT* tests

package services

import (
	"math"
	"testing"
)

func TestMATSourceStepDoesNotReset(t *testing.T) {
	state := NewMATNeuron()
	state.V = 20.0
	want := state.V + state.Dt*(-state.V)/state.TauM
	if got := state.Step(0.0); got != 1 {
		t.Fatalf("event = %d, want 1", got)
	}
	if state.V != want || state.Theta1 != 37.0 || state.Theta2 != 2.0 {
		t.Fatalf("non-resetting state mismatch: %+v", *state)
	}
}

func TestMATSourceExactDecayAndAtomicFailure(t *testing.T) {
	state := NewMATNeuron()
	state.Theta1, state.Theta2, state.RefractoryRemaining = 7.0, 4.0, 1.25
	want1 := state.Theta1 * math.Exp(-state.Dt/state.Tau1)
	want2 := state.Theta2 * math.Exp(-state.Dt/state.Tau2)
	if got := state.Step(0.42); got != 0 || state.Theta1 != want1 || state.Theta2 != want2 {
		t.Fatalf("source step mismatch: event=%d state=%+v", got, *state)
	}
	before := *state
	if got := state.Step(math.NaN()); got != -1 || *state != before {
		t.Fatalf("invalid step was not atomic: event=%d state=%+v", got, *state)
	}
}

func BenchmarkMATSource(b *testing.B) {
	state := NewMATNeuron()
	spikes := 0
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		result := state.Step(0.5)
		if result < 0 {
			b.Fatal("invalid source MAT step")
		}
		spikes += result
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
