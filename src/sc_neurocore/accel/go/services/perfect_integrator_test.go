// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for the Perfect Integrator

package services

import (
	"math"
	"testing"
)

func TestPerfectIntegratorSourceGoldenEventCounts(t *testing.T) {
	goldens := []struct {
		current float64
		spikes  int
	}{
		{0.0, 0},
		{0.333, 32},
		{0.7, 66},
		{2.0, 200},
		{3.0, 250},
		{5.0, 500},
		{20.0, 1000},
	}
	for _, golden := range goldens {
		state := NewPerfectIntegratorNeuron()
		spikes := 0
		for index := 0; index < 1000; index++ {
			spike, err := state.Step(golden.current)
			if err != nil {
				t.Fatalf("current=%v: unexpected error: %v", golden.current, err)
			}
			spikes += spike
		}
		if spikes != golden.spikes {
			t.Fatalf("current=%v: got %d spikes, want %d", golden.current, spikes, golden.spikes)
		}
	}
}

func TestPerfectIntegratorCompleteContract(t *testing.T) {
	initial := PerfectIntegratorNeuronState{
		V:          0.25,
		CM:         1.7,
		VThreshold: 1.3,
		VReset:     -0.2,
		Dt:         0.37,
	}
	trace, spikes, err := SimulatePerfectIntegratorTrace(initial, 300, 2.2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(trace) != 300 || spikes != 75 {
		t.Fatalf("got len=%d spikes=%d, want len=300 spikes=75", len(trace), spikes)
	}
	if trace[len(trace)-1] != 0.2788235294117647 {
		t.Fatalf("final voltage mismatch: %.17g", trace[len(trace)-1])
	}

	empty, emptySpikes, err := SimulatePerfectIntegratorTrace(initial, 0, 2.2)
	if err != nil || len(empty) != 0 || emptySpikes != 0 {
		t.Fatalf("empty trace contract failed: len=%d spikes=%d err=%v", len(empty), emptySpikes, err)
	}
}

func TestPerfectIntegratorRejectsInvalidWorkBeforeMutation(t *testing.T) {
	state := NewPerfectIntegratorNeuron()
	state.V = 0.25
	before := state.V
	if _, err := state.Step(math.Inf(1)); err == nil {
		t.Fatal("expected non-finite current to fail")
	}
	if state.V != before {
		t.Fatalf("invalid current mutated voltage to %.17g", state.V)
	}

	state.VThreshold = 1.0e308
	state.CM = 1.0e-308
	if _, err := state.Step(1.0e308); err == nil {
		t.Fatal("expected non-finite candidate to fail")
	}
	if state.V != before {
		t.Fatalf("invalid candidate mutated voltage to %.17g", state.V)
	}

	if trace, spikes, err := SimulatePerfectIntegratorTrace(*state, -1, 0.0); err == nil || trace != nil || spikes != 0 {
		t.Fatalf("negative run did not fail closed: trace=%v spikes=%d err=%v", trace, spikes, err)
	}
}

func TestPerfectIntegratorResetPreservesParameters(t *testing.T) {
	state := PerfectIntegratorNeuronState{
		V:          0.5,
		CM:         2.0,
		VThreshold: 3.0,
		VReset:     -1.0,
		Dt:         0.05,
	}
	state.Reset()
	if state.V != -1.0 || state.CM != 2.0 || state.VThreshold != 3.0 || state.Dt != 0.05 {
		t.Fatalf("reset corrupted parameters: %+v", state)
	}
}

func BenchmarkPerfectIntegratorEuler(b *testing.B) {
	state := NewPerfectIntegratorNeuron()
	spikes := 0
	for index := 0; index < b.N; index++ {
		spike, err := state.Step(5.0)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if !finitePerfectIntegrator(state.V) || spikes < 0 {
		b.Fatalf("invalid final state: v=%v spikes=%d", state.V, spikes)
	}
}
