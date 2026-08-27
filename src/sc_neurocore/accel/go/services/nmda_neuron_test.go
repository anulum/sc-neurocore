// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go NMDA source and retained-SC contracts

package services

import (
	"math"
	"testing"
)

func TestNMDASourceAnchorAndAtomicFailure(t *testing.T) {
	state := NewNMDANeuron()
	event, err := state.TryStep(0.3)
	if err != nil || event != 0 || math.Abs(state.V-(-69.9700375)) > 1e-12 {
		t.Fatalf("source anchor mismatch: event=%d state=%+v err=%v", event, state, err)
	}
	before := *state
	if _, err = state.TryStep(math.NaN()); err == nil || *state != before {
		t.Fatal("invalid current must fail atomically")
	}
}

func TestSCWBNMDARetainedAnchor(t *testing.T) {
	state := NewSCWBNMDAMagnesiumBlockNeuron()
	event, err := state.TryStep(5)
	if err != nil || event != 0 || math.Abs(state.V-(-63.15566378039578)) > 1e-12 ||
		math.Abs(state.SNmda-0.025) > 1e-15 {
		t.Fatalf("retained anchor mismatch: event=%d state=%+v err=%v", event, state, err)
	}
}

func BenchmarkNMDASourceRK2(b *testing.B) {
	state := NewNMDANeuron()
	for i := 0; i < b.N; i++ {
		state.Step(0.6)
	}
}

func BenchmarkSCWBNMDAMagnesiumBlock(b *testing.B) {
	state := NewSCWBNMDAMagnesiumBlockNeuron()
	for i := 0; i < b.N; i++ {
		state.Step(5)
	}
}
