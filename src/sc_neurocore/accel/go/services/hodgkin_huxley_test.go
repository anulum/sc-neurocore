// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Hodgkin-Huxley service tests

package services

import (
	"math"
	"testing"
)

// hhReferenceOpeningRate is an independent transcription of the singular-limit opening rate
// so the cross-check does not reuse the kernel's own helpers.
func hhReferenceOpeningRate(scale, shift, denom, limit, v float64) float64 {
	d := v + shift
	if math.Abs(d) < 1e-7 {
		return limit
	}
	return scale * d / (1.0 - math.Exp(-d/denom))
}

// hhReferenceStep is an independent baseline-Euler macro step used only to pin the kernel's
// arithmetic ordering (gates first, then voltage on the fresh gates).
func hhReferenceStep(s HodgkinHuxleyNeuronState, current float64) hodgkinHuxleyState {
	st := hodgkinHuxleyState{v: s.V, m: s.M, h: s.H, n: s.N}
	substeps := int(math.Round(1.0 / s.Dt))
	for i := 0; i < substeps; i++ {
		am := hhReferenceOpeningRate(0.1, 40.0, 10.0, 1.0, st.v)
		bm := 4.0 * math.Exp(-(st.v+65.0)/18.0)
		ah := 0.07 * math.Exp(-(st.v+65.0)/20.0)
		bh := 1.0 / (1.0 + math.Exp(-(st.v+35.0)/10.0))
		an := hhReferenceOpeningRate(0.01, 55.0, 10.0, 0.1, st.v)
		bn := 0.125 * math.Exp(-(st.v+65.0)/80.0)
		st.m += (am*(1.0-st.m) - bm*st.m) * s.Dt
		st.h += (ah*(1.0-st.h) - bh*st.h) * s.Dt
		st.n += (an*(1.0-st.n) - bn*st.n) * s.Dt
		iNa := s.GNa * math.Pow(st.m, 3.0) * st.h * (st.v - s.ENa)
		iK := s.GK * math.Pow(st.n, 4.0) * (st.v - s.EK)
		iL := s.GL * (st.v - s.EL)
		st.v += (-iNa - iK - iL + current) / s.CM * s.Dt
	}
	return st
}

// TestHodgkinHuxleyMatchesIndependentEuler pins the kernel's single-macro-step state to an
// independent baseline-Euler transcription, guarding against a gate/voltage ordering regression.
func TestHodgkinHuxleyMatchesIndependentEuler(t *testing.T) {
	neuron := NewHodgkinHuxleyNeuron()
	neuron.V = -60.0
	neuron.M = 0.06
	neuron.H = 0.55
	neuron.N = 0.35
	expected := hhReferenceStep(*neuron, 8.5)

	spike, err := neuron.Step(8.5)
	if err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	if math.Abs(neuron.V-expected.v) > 1e-10 || math.Abs(neuron.M-expected.m) > 1e-10 ||
		math.Abs(neuron.H-expected.h) > 1e-10 || math.Abs(neuron.N-expected.n) > 1e-10 {
		t.Fatalf("state mismatch got (%g,%g,%g,%g) expected (%g,%g,%g,%g)",
			neuron.V, neuron.M, neuron.H, neuron.N, expected.v, expected.m, expected.h, expected.n)
	}
}

func TestHodgkinHuxleyInvalidCurrentPreservesState(t *testing.T) {
	neuron := NewHodgkinHuxleyNeuron()
	before := *neuron
	if _, err := neuron.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if *neuron != before {
		t.Fatalf("state mutated after invalid current")
	}
}

func TestHodgkinHuxleyCorruptStatePreservesState(t *testing.T) {
	neuron := NewHodgkinHuxleyNeuron()
	neuron.H = math.Inf(1)
	before := *neuron
	if _, err := neuron.Step(6.0); err == nil {
		t.Fatal("expected invalid state error")
	}
	if *neuron != before {
		t.Fatalf("state mutated after invalid state")
	}
}

// TestHodgkinHuxleyMatchesPythonGolden pins the Go kernel to the Python golden
// (models/hodgkin_huxley.py HodgkinHuxleyNeuron with the default baseline_euler integrator,
// 100 explicit-Euler sub-steps per macro step): silent at zero drive, six action potentials at
// I=10 over 100 macro steps, and nine at I=20. Hodgkin-Huxley gating uses exp, so the trace is
// not bit-exact across libms; the spike count is the stable observable and is the parity contract
// — not a "spike is 0 or 1" smoke test. The Rust and Julia kernels reproduce the same counts.
func TestHodgkinHuxleyMatchesPythonGolden(t *testing.T) {
	for _, tc := range []struct {
		current float64
		want    int
	}{
		{0.0, 0},
		{10.0, 6},
		{20.0, 9},
	} {
		_, spikes := SimulateHodgkinHuxleyNeuron(100, tc.current)
		if spikes != tc.want {
			t.Fatalf("Hodgkin-Huxley Go kernel must reproduce the Python golden at I=%g over 100 macro steps: want %d spikes, got %d", tc.current, tc.want, spikes)
		}
	}
}

// BenchmarkHodgkinHuxleyStep measures one macro step (100 explicit-Euler sub-steps of the
// four-state model) so the Go backend carries an honest, runnable per-step timing.
func BenchmarkHodgkinHuxleyStep(b *testing.B) {
	s := NewHodgkinHuxleyNeuron()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Step(10.0); err != nil {
			s = NewHodgkinHuxleyNeuron()
		}
	}
}
