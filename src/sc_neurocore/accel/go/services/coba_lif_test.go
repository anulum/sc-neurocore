// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for COBA LIF RK4 service

package services

import (
	"math"
	"testing"
)

func TestCOBALIFRK4ConductanceInjection(t *testing.T) {
	state := NewCOBALIFNeuron()
	vCandidate, geCandidate, giCandidate := state.rk4Candidate(state.V, 5.0, 3.0, 0.0)

	spike, err := state.StepWithConductance(0.0, 5.0, 3.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-vCandidate) > 1.0e-12 || math.Abs(state.GE-geCandidate) > 1.0e-12 || math.Abs(state.GI-giCandidate) > 1.0e-12 {
		t.Fatalf("RK4 candidate mismatch: got (%g,%g,%g), want (%g,%g,%g)", state.V, state.GE, state.GI, vCandidate, geCandidate, giCandidate)
	}
}

func TestCOBALIFFactoryMatchesBretteBenchmarkOne(t *testing.T) {
	state := NewCOBALIFNeuron()
	want := COBALIFNeuronState{V: -60.0, GE: 0.0, GI: 0.0, RefractoryTime: 0.0, CM: 200.0, GL: 10.0, EL: -60.0, EE: 0.0, EI: -80.0, TauE: 5.0, TauI: 10.0, VThreshold: -50.0, VReset: -60.0, RefractoryPeriod: 5.0, Dt: 0.1}
	if *state != want {
		t.Fatalf("factory mismatch: got %+v want %+v", *state, want)
	}
}

func TestCOBALIFInvalidStateDoesNotMutate(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.GE = -1.0
	before := *state
	if _, err := state.Step(0.0); err == nil {
		t.Fatal("expected invalid conductance error")
	}
	if *state != before {
		t.Fatalf("invalid state mutated: got %+v want %+v", *state, before)
	}
}

func TestCOBALIFSuprathresholdResetPreservesConductanceCandidate(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.V = -51.0
	_, geCandidate, _ := state.rk4Candidate(state.V, 5.0, 0.0, 1.0e5)

	spike, err := state.StepWithConductance(1.0e5, 5.0, 0.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if state.V != state.VReset {
		t.Fatalf("voltage did not reset: got %g want %g", state.V, state.VReset)
	}
	if math.Abs(state.GE-geCandidate) > 1.0e-12 {
		t.Fatalf("conductance candidate not preserved: got %g want %g", state.GE, geCandidate)
	}
	if state.RefractoryTime != state.RefractoryPeriod {
		t.Fatalf("refractory timer not loaded: got %g want %g", state.RefractoryTime, state.RefractoryPeriod)
	}
}

func TestCOBALIFRefractoryIntervalClampsWithoutFloatResidue(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.V = -51.0
	state.EL = -65.0
	if spike, err := state.Step(1.0e5); err != nil || spike != 1 {
		t.Fatalf("expected source spike, got spike=%d err=%v", spike, err)
	}
	for step := 0; step < 50; step++ {
		spike, err := state.Step(0.0)
		if err != nil || spike != 0 || state.V != state.VReset {
			t.Fatalf("refractory hold failed at step %d: state=%+v spike=%d err=%v", step, *state, spike, err)
		}
	}
	if state.RefractoryTime != 0.0 {
		t.Fatalf("refractory timer retained residue: %g", state.RefractoryTime)
	}
	if _, err := state.Step(0.0); err != nil || !(state.V < state.VReset) {
		t.Fatalf("integration did not resume after exact hold: state=%+v err=%v", *state, err)
	}
}

func TestCOBALIFRawVoltageCandidateRejectsBeforeReset(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.V = 90.0
	before := *state
	if _, err := state.Step(1.0e8); err == nil {
		t.Fatal("expected raw voltage candidate rejection")
	}
	if *state != before {
		t.Fatalf("rejected candidate mutated state: got %+v want %+v", *state, before)
	}
}

func BenchmarkCOBALIFRK4(b *testing.B) {
	const current = 650.0
	const deltaGE = 0.15
	const deltaGI = 0.07
	state := NewCOBALIFNeuron()
	spikes := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := state.StepWithConductance(current, deltaGE, deltaGI)
		if err != nil {
			b.Fatalf("invalid RK4 step: %v", err)
		}
		spikes += result
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
