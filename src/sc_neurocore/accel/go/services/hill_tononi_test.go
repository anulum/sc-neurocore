// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Hill-Tononi source fidelity tests

package services

import (
	"math"
	"testing"
)

func TestHillTononiSourceDefaultsAndPythonAnchor(t *testing.T) {
	s := NewHillTononiNeuron()
	if s.V != -70.0 || s.Theta != -51.0 || s.DK != 0.001 || s.Dt != 0.25 {
		t.Fatalf("unexpected source defaults: %+v", s)
	}
	if spike := s.Step(12.0); spike != 0 {
		t.Fatalf("unexpected first-step spike: %d", spike)
	}
	expected := [6]float64{
		-69.81228106951788, -51.0, 0.0010000391293823398,
		0.28718473563652219, 0.14517852005930809, 0.037318086618308974,
	}
	observed := [6]float64{s.V, s.Theta, s.DK, s.MH, s.MT, s.HT}
	for index := range expected {
		if math.Abs(observed[index]-expected[index]) > 2e-12 {
			t.Fatalf("state[%d] = %.17g, want %.17g", index, observed[index], expected[index])
		}
	}
}

func TestHillTononiDynamicThresholdAndPulse(t *testing.T) {
	s := NewHillTononiNeuron()
	s.V, s.Theta = -50.0, -51.0
	spike, err := s.TryStep(0.0)
	if err != nil || spike != 1 {
		t.Fatalf("spike = %d, err = %v", spike, err)
	}
	if s.V != 30.0 || s.Theta != 30.0 || s.SpikeTimer != 2.0 {
		t.Fatalf("unexpected spike state: %+v", s)
	}
	if spike, err = s.TryStep(0.0); err != nil || spike != 0 {
		t.Fatalf("refractory spike = %d, err = %v", spike, err)
	}
	if s.SpikeTimer != 1.75 || s.V >= 30.0 {
		t.Fatalf("post-spike pulse did not advance: %+v", s)
	}
}

func TestHillTononiInvalidStepIsAtomic(t *testing.T) {
	s := NewHillTononiNeuron()
	before := [6]float64{s.V, s.Theta, s.DK, s.MH, s.MT, s.HT}
	if _, err := s.TryStep(math.NaN()); err == nil {
		t.Fatal("non-finite current was accepted")
	}
	after := [6]float64{s.V, s.Theta, s.DK, s.MH, s.MT, s.HT}
	if before != after {
		t.Fatalf("state mutated on error: before=%v after=%v", before, after)
	}
}

func TestHillTononiCrossBackendSpikeAnchor(t *testing.T) {
	_, spikes := SimulateHillTononiNeuron(200000, 20.0)
	if spikes != 538 {
		t.Fatalf("spikes = %d, want 538", spikes)
	}
}

func TestHillTononiCorticalWakeIsSilentWithoutInput(t *testing.T) {
	_, spikes := SimulateHillTononiNeuron(10000, 0.0)
	if spikes != 0 {
		t.Fatalf("spikes = %d, want 0", spikes)
	}
}

func TestHillTononiOptionalCurrentsRemainFinite(t *testing.T) {
	s := NewHillTononiNeuron()
	s.GH, s.GT = 1.0, 1.0
	for step := 0; step < 200; step++ {
		if _, err := s.TryStep(0.0); err != nil {
			t.Fatal(err)
		}
	}
	if !hillTononiFinite(s.V, s.MH, s.MT, s.HT) {
		t.Fatalf("optional-current state is non-finite: %+v", s)
	}
}

func BenchmarkHillTononiRK4(b *testing.B) {
	s := NewHillTononiNeuron()
	spikes := 0
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		spikes += s.Step(20.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
