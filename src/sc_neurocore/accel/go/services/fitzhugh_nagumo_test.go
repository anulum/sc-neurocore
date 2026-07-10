// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for fitzhugh_nagumo

package services

import (
	"math"
	"testing"
)

func TestFitzHughNagumoNew(t *testing.T) {
	s := NewFitzHughNagumoNeuron()
	if !s.Valid() {
		t.Fatal("default FitzHugh-Nagumo state is invalid")
	}
}

func TestFitzHughNagumoStep(t *testing.T) {
	s := NewFitzHughNagumoNeuron()
	spike, err := s.Step(0.5)
	if err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	if !s.Valid() {
		t.Fatal("state became invalid after a step")
	}
}

// TestFitzHughNagumoMatchesPythonGolden pins the Go services kernel to the Python
// golden (models/fitzhugh_nagumo.py FitzHughNagumoNeuron.simulate, RK4). Two
// regimes: a single action potential under a strong drive (I=10, 100 steps) and a
// clean five-spike partial train on the limit cycle (I=0.5, 2000 steps). The RK4
// right-hand side is exact arithmetic (v*v*v, no transcendentals), so the final v
// is bit-identical to the NumPy reference — this is the parity contract, not a
// "spike is 0 or 1" smoke test.
func TestFitzHughNagumoMatchesPythonGolden(t *testing.T) {
	single, singleSpikes := SimulateFitzHughNagumoNeuron(100, 10.0)
	if singleSpikes != 1 {
		t.Fatalf("FitzHugh-Nagumo Go kernel must reproduce the Python golden (1 AP @ I=10, 100 steps); got %d", singleSpikes)
	}
	if single[99] != 3.224001034533657 {
		t.Fatalf("final v must match the NumPy reference bit-for-bit; got %v, want 3.224001034533657", single[99])
	}

	train, trainSpikes := SimulateFitzHughNagumoNeuron(2000, 0.5)
	if trainSpikes != 5 {
		t.Fatalf("FitzHugh-Nagumo Go kernel must reproduce the Python golden (5 AP @ I=0.5, 2000 steps); got %d", trainSpikes)
	}
	if train[1999] != -0.18831885872309226 {
		t.Fatalf("final v must match the NumPy reference bit-for-bit; got %v, want -0.18831885872309226", train[1999])
	}
}

func TestFitzHughNagumoSilentAtZeroCurrent(t *testing.T) {
	_, spikes := SimulateFitzHughNagumoNeuron(100, 0.0)
	if spikes != 0 {
		t.Fatalf("FitzHugh-Nagumo should be silent at zero drive; got %d spikes", spikes)
	}
}

func TestFitzHughNagumoInvalidCurrentPreservesState(t *testing.T) {
	s := NewFitzHughNagumoNeuron()
	before := *s
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if *s != before {
		t.Fatal("state mutated after invalid current")
	}
}

func TestFitzHughNagumoCorruptStatePreservesState(t *testing.T) {
	s := NewFitzHughNagumoNeuron()
	s.V = math.Inf(1)
	before := *s
	if _, err := s.Step(0.5); err == nil {
		t.Fatal("expected invalid state error")
	}
	if *s != before {
		t.Fatal("state mutated after invalid state")
	}
}

// BenchmarkFitzHughNagumoStep measures one RK4 step so the Go backend carries an
// honest, runnable per-step timing rather than an unmeasured claim.
func BenchmarkFitzHughNagumoStep(b *testing.B) {
	s := NewFitzHughNagumoNeuron()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Step(0.5); err != nil {
			s = NewFitzHughNagumoNeuron()
		}
	}
}
