// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for srm0

package services

import (
	"math"
	"testing"
)

func srm0ReferenceStep(s *SRM0NeuronState, current float64) (float64, float64) {
	membraneDecay := math.Exp(-s.Dt / s.TauM)
	etaDecay := math.Exp(-s.Dt / s.TauEta)
	rateDelta := (1.0 / s.TauM) - (1.0 / s.TauEta)
	var etaCoupling float64
	if math.Abs(rateDelta) < 1.0e-14 {
		etaCoupling = s.Dt * membraneDecay / s.TauM
	} else {
		etaCoupling = (etaDecay - membraneDecay) / (s.TauM * rateDelta)
	}
	steady := s.VRest + s.Resistance*current
	return steady + (s.V-steady)*membraneDecay + s.Eta*etaCoupling, s.Eta * etaDecay
}

func TestSRM0StepMatchesExactFlow(t *testing.T) {
	state := NewSRM0Neuron()
	state.Eta = -2.0
	wantV, wantEta := srm0ReferenceStep(state, 0.5)

	if got := state.Step(0.5); got != 0 {
		t.Fatalf("subthreshold exact-flow step spiked: %d", got)
	}
	if math.Abs(state.V-wantV) > 1e-12 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, wantV)
	}
	if math.Abs(state.Eta-wantEta) > 1e-12 {
		t.Fatalf("eta mismatch: got %.17g want %.17g", state.Eta, wantEta)
	}
}

func TestSRM0RejectsInvalidCurrentWithoutMutation(t *testing.T) {
	state := NewSRM0Neuron()
	v0, eta0, t0 := state.V, state.Eta, state.T
	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid current must fail closed, got %d", got)
	}
	if state.V != v0 || state.Eta != eta0 || state.T != t0 {
		t.Fatalf("invalid current mutated state: got (%v, %v, %v)", state.V, state.Eta, state.T)
	}
}

func BenchmarkSRM0ExactFlow(b *testing.B) {
	state := NewSRM0Neuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		result := state.Step(2.0)
		if result < 0 {
			b.Fatalf("invalid exact-flow step at iteration %d", i)
		}
		spikes += result
	}
	b.ReportMetric(float64(spikes), "spikes")
}
