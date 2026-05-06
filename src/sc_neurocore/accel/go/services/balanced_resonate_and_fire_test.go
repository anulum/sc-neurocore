// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go BRF service tests

package services

import (
	"math"
	"testing"
)

func TestBalancedRFSustainOscillationBoundary(t *testing.T) {
	got, err := BalancedRFSustainOscillationBoundary(10.0, 0.01)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := (-1.0 + math.Sqrt(1.0-0.1*0.1)) / 0.01
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("boundary mismatch: got %.16f want %.16f", got, want)
	}
}

func TestBalancedRFStepUpdatesRefractoryAfterSpike(t *testing.T) {
	neuron := NewBalancedResonateAndFireNeuron()
	spike, err := neuron.Step(200.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if neuron.Q <= 0.0 {
		t.Fatalf("expected refractory state to increase, got %.16f", neuron.Q)
	}
	if neuron.DynamicThreshold() <= neuron.Threshold {
		t.Fatalf("expected refractory threshold above base threshold")
	}
}

func TestBalancedRFRejectsInvalidBoundary(t *testing.T) {
	if _, err := BalancedRFSustainOscillationBoundary(200.0, 0.01); err == nil {
		t.Fatalf("expected invalid dt * omega boundary to fail")
	}
}

func BenchmarkBalancedRFStep(b *testing.B) {
	neuron := NewBalancedResonateAndFireNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		spike, err := neuron.Step(2.0)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	_ = spikes
}
