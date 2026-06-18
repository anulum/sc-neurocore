// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for spike_response

package services

import (
	"math"
	"testing"
)

func spikeResponseReferenceV(s *SpikeResponseNeuronState, weightedInput float64) float64 {
	eta := 0.0
	if s.TimeSinceSpike < 100.0 {
		eta = s.EtaReset * math.Exp(-s.TimeSinceSpike/s.TauEta)
	}
	kappa := weightedInput * (1.0 - math.Exp(-s.Dt/s.TauKappa))
	return eta + kappa
}

func TestSpikeResponseStepMatchesKernel(t *testing.T) {
	state := NewSpikeResponseNeuron()
	want := spikeResponseReferenceV(state, 5.0)

	if got := state.Step(5.0); got != 0 {
		t.Fatalf("subthreshold kernel step spiked: %d", got)
	}
	if math.Abs(state.V-want) > 1e-12 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, want)
	}
	if state.TimeSinceSpike != 1001.0 {
		t.Fatalf("time_since_spike mismatch: got %.17g", state.TimeSinceSpike)
	}
}

func TestSpikeResponseSpikeResetsClock(t *testing.T) {
	state := NewSpikeResponseNeuron()
	if got := state.Step(10.0); got != 1 {
		t.Fatalf("suprathreshold kernel step did not spike: %d", got)
	}
	if state.V != 0.0 || state.TimeSinceSpike != 0.0 {
		t.Fatalf("spike reset mismatch: got (%v, %v)", state.V, state.TimeSinceSpike)
	}
}

func TestSpikeResponseRejectsInvalidInputWithoutMutation(t *testing.T) {
	state := NewSpikeResponseNeuron()
	v0, t0 := state.V, state.TimeSinceSpike
	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid input must fail closed, got %d", got)
	}
	if state.V != v0 || state.TimeSinceSpike != t0 {
		t.Fatalf("invalid input mutated state: got (%v, %v)", state.V, state.TimeSinceSpike)
	}
}

func BenchmarkSpikeResponseKernel(b *testing.B) {
	state := NewSpikeResponseNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		result := state.Step(10.0)
		if result < 0 {
			b.Fatalf("invalid kernel step at iteration %d", i)
		}
		spikes += result
	}
	b.ReportMetric(float64(spikes), "spikes")
}
