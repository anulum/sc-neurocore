// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for composite reduced adaptive-threshold IF

package services

import (
	"math"
	"testing"
)

func adaptiveThresholdExactRelaxation(state, steadyState, tau, dt float64) float64 {
	return steadyState + (state-steadyState)*math.Exp(-dt/tau)
}

func TestAdaptiveThresholdIFCatalogueDefaults(t *testing.T) {
	state := NewAdaptiveThresholdIFNeuron()
	if !state.Valid() {
		t.Fatal("catalogue defaults must be valid")
	}
	if state.V != -65.0 || state.Theta != -50.0 || state.DeltaTheta != 5.0 ||
		state.TauM != 10.0 || state.TauTheta != 50.0 || state.Dt != 0.1 {
		t.Fatalf("unexpected catalogue defaults: %+v", state)
	}
}

func TestAdaptiveThresholdIFExactRelaxationNoSpike(t *testing.T) {
	state := &AdaptiveThresholdIFNeuronState{
		V: -60.0, Theta: -52.0, VRest: -70.0, VReset: -68.0, ThetaRest: -48.0,
		DeltaTheta: 3.0, TauM: 8.0, TauTheta: 40.0, Dt: 0.05,
	}
	expectedV := adaptiveThresholdExactRelaxation(state.V, state.VRest+12.5, state.TauM, state.Dt)
	expectedTheta := adaptiveThresholdExactRelaxation(state.Theta, state.ThetaRest, state.TauTheta, state.Dt)
	spike, err := state.Step(12.5)
	if err != nil || spike != 0 {
		t.Fatalf("unexpected result: spike=%d error=%v", spike, err)
	}
	if math.Abs(state.V-expectedV) > 1.0e-12 || math.Abs(state.Theta-expectedTheta) > 1.0e-12 {
		t.Fatalf("exact-relaxation mismatch: got (%0.17g, %0.17g)", state.V, state.Theta)
	}
}

func TestAdaptiveThresholdIFCrossingResetAndFixedShift(t *testing.T) {
	state := &AdaptiveThresholdIFNeuronState{
		V: -50.5, Theta: -51.0, VRest: -65.0, VReset: -65.0, ThetaRest: -50.0,
		DeltaTheta: 5.0, TauM: 10.0, TauTheta: 50.0, Dt: 0.1,
	}
	spike, err := state.Step(0.0)
	if err != nil || spike != 1 {
		t.Fatalf("expected one spike, got spike=%d error=%v", spike, err)
	}
	if state.V != -65.0 {
		t.Fatalf("wrong membrane reset: %v", state.V)
	}
	relaxed := adaptiveThresholdExactRelaxation(-51.0, -50.0, 50.0, 0.1)
	if math.Abs(state.Theta-(relaxed+5.0)) > 1.0e-12 {
		t.Fatalf("wrong threshold shift: got %v, want %v", state.Theta, relaxed+5.0)
	}
	spike, err = state.Step(0.0)
	if err != nil || spike != 0 {
		t.Fatalf("shifted threshold must not retrigger: spike=%d error=%v", spike, err)
	}
}

func TestAdaptiveThresholdIFSilentBelowThreshold(t *testing.T) {
	state := NewAdaptiveThresholdIFNeuron()
	spikes := 0
	for i := 0; i < 500; i++ {
		spike, err := state.Step(0.0)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if spikes != 0 {
		t.Fatalf("zero drive must stay silent: %d spikes", spikes)
	}
}

func TestAdaptiveThresholdIFInvalidStatePreservesState(t *testing.T) {
	state := NewAdaptiveThresholdIFNeuron()
	state.V = -60.0
	state.Theta = -55.0
	beforeV, beforeTheta := state.V, state.Theta
	state.Dt = 0.0
	if _, err := state.Step(0.5); err == nil {
		t.Fatal("expected invalid-state error")
	}
	if state.V != beforeV || state.Theta != beforeTheta {
		t.Fatalf("state mutated on invalid input: got (%v, %v)", state.V, state.Theta)
	}
}

func TestAdaptiveThresholdIFNonFiniteUpdatePreservesState(t *testing.T) {
	state := NewAdaptiveThresholdIFNeuron()
	state.V = -math.MaxFloat64
	beforeV, beforeTheta := state.V, state.Theta
	if _, err := state.Step(math.MaxFloat64); err == nil {
		t.Fatal("expected non-finite update error")
	}
	if state.V != beforeV || state.Theta != beforeTheta {
		t.Fatalf("state mutated on non-finite update: got (%v, %v)", state.V, state.Theta)
	}
}

func TestAdaptiveThresholdIFLegacySimulationRejectsInvalidArguments(t *testing.T) {
	trace, spikes := SimulateAdaptiveThresholdIFNeuron(-1, 0.0)
	if trace != nil || spikes != 0 {
		t.Fatalf("negative step count must fail closed: trace=%v spikes=%d", trace, spikes)
	}
	trace, spikes = SimulateAdaptiveThresholdIFNeuron(1, math.NaN())
	if trace != nil || spikes != 0 {
		t.Fatalf("non-finite current must fail closed: trace=%v spikes=%d", trace, spikes)
	}
}
