// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import (
	"math"
	"testing"
)

func TestWangBuzsakiNew(t *testing.T) {
	s := NewWangBuzsakiNeuron()
	if !ValidateWangBuzsaki(s) {
		t.Fatal("default Wang-Buzsaki state is invalid")
	}
}

func TestWangBuzsakiStep(t *testing.T) {
	s := NewWangBuzsakiNeuron()
	spike, err := s.Step(10.0)
	if err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	if !ValidateWangBuzsaki(s) {
		t.Fatal("state became invalid after a step")
	}
}

// TestWangBuzsakiMatchesPythonGolden pins the Go kernel to the Python golden
// (models/wang_buzsaki.py WangBuzsakiNeuron): three action potentials at I=10 over 20 macro
// steps — the same count the gauss_seidel schema runner and the Q16.16 RTL reproduce three-way
// exactly. It is the parity contract for this kernel, not a mere "spike is 0 or 1" smoke test.
func TestWangBuzsakiMatchesPythonGolden(t *testing.T) {
	_, spikes := SimulateWangBuzsakiNeuron(20, 10.0)
	if spikes != 3 {
		t.Fatalf("Wang-Buzsaki Go kernel must reproduce the Python golden (3 AP @ I=10, 20 macro steps); got %d", spikes)
	}
}

func TestWangBuzsakiSilentAtZeroCurrent(t *testing.T) {
	_, spikes := SimulateWangBuzsakiNeuron(20, 0.0)
	if spikes != 0 {
		t.Fatalf("Wang-Buzsaki should be silent at zero current; got %d spikes", spikes)
	}
}

func TestWangBuzsakiInvalidCurrentPreservesState(t *testing.T) {
	s := NewWangBuzsakiNeuron()
	before := *s
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if *s != before {
		t.Fatal("state mutated after invalid current")
	}
}

func TestWangBuzsakiCorruptStatePreservesState(t *testing.T) {
	s := NewWangBuzsakiNeuron()
	s.H = math.Inf(1)
	before := *s
	if _, err := s.Step(10.0); err == nil {
		t.Fatal("expected invalid state error")
	}
	if *s != before {
		t.Fatal("state mutated after invalid state")
	}
}

// BenchmarkWangBuzsakiStep measures one 0.5 ms macro step (50 sequential sub-steps) so the
// Go backend carries an honest, runnable per-step timing rather than an unmeasured claim.
func BenchmarkWangBuzsakiStep(b *testing.B) {
	s := NewWangBuzsakiNeuron()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Step(10.0); err != nil {
			s = NewWangBuzsakiNeuron()
		}
	}
}
