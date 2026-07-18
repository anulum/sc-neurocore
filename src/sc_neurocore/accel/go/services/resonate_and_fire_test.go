// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Izhikevich resonate-and-fire

package services

import (
	"math"
	"testing"
)

func resonateExactFlow(x, y, current, b, omega, dt float64) (float64, float64) {
	denominator := b*b + omega*omega
	xSS := -b * current / denominator
	ySS := omega * current / denominator
	dx := x - xSS
	dy := y - ySS
	decay := math.Exp(b * dt)
	angle := omega * dt
	return xSS + decay*(dx*math.Cos(angle)-dy*math.Sin(angle)),
		ySS + decay*(dx*math.Sin(angle)+dy*math.Cos(angle))
}

func TestResonateAndFireSourceDefaults(t *testing.T) {
	state := NewResonateAndFireNeuron()
	if !state.Valid() {
		t.Fatal("source defaults must be valid")
	}
	if state.B != -1.0 || state.Omega != 10.0 || state.Threshold != 1.0 || state.Dt != 0.01 {
		t.Fatalf("unexpected source defaults: %+v", state)
	}
}

func TestResonateAndFireExactFlowNoSpike(t *testing.T) {
	state := &ResonateAndFireNeuronState{
		X: 0.3, Y: -0.2, B: -0.2, Omega: 1.7, Threshold: 100.0, Dt: 1.25,
	}
	expectedX, expectedY := resonateExactFlow(
		state.X, state.Y, 0.8, state.B, state.Omega, state.Dt,
	)
	spike, err := state.Step(0.8)
	if err != nil || spike != 0 {
		t.Fatalf("unexpected result: spike=%d error=%v", spike, err)
	}
	if math.Abs(state.X-expectedX) > 1.0e-12 || math.Abs(state.Y-expectedY) > 1.0e-12 {
		t.Fatalf("exact-flow mismatch: got (%0.17g, %0.17g)", state.X, state.Y)
	}
}

func TestResonateAndFireVoltageCrossingAndSourceReset(t *testing.T) {
	state := &ResonateAndFireNeuronState{
		X: 0.0, Y: 0.99, B: 0.0, Omega: 1.0, Threshold: 1.0, Dt: 0.1,
	}
	spike, err := state.Step(10.0)
	if err != nil || spike != 1 {
		t.Fatalf("expected one spike, got spike=%d error=%v", spike, err)
	}
	if state.X != 0.0 || state.Y != 1.0 {
		t.Fatalf("wrong source reset: (%v, %v)", state.X, state.Y)
	}
	spike, err = state.Step(0.0)
	if err != nil || spike != 0 {
		t.Fatalf("reset threshold must not retrigger: spike=%d error=%v", spike, err)
	}
}

func TestResonateAndFireRadiusAloneDoesNotSpike(t *testing.T) {
	state := &ResonateAndFireNeuronState{
		X: 2.0, Y: 0.0, B: 0.0, Omega: 1.0, Threshold: 1.0, Dt: 0.01,
	}
	spike, err := state.Step(0.0)
	if err != nil || spike != 0 {
		t.Fatalf("radius is not the event surface: spike=%d error=%v", spike, err)
	}
}

func TestResonateAndFireInvalidStatePreservesState(t *testing.T) {
	state := NewResonateAndFireNeuron()
	state.X = 0.25
	state.Y = -0.5
	beforeX, beforeY := state.X, state.Y
	state.Dt = 0.0
	if _, err := state.Step(0.5); err == nil {
		t.Fatal("expected invalid-state error")
	}
	if state.X != beforeX || state.Y != beforeY {
		t.Fatalf("state mutated on invalid input: got (%v, %v)", state.X, state.Y)
	}
}

func TestResonateAndFireLegacySimulationRejectsInvalidArguments(t *testing.T) {
	trace, spikes := SimulateResonateAndFireNeuron(-1, 0.0)
	if trace != nil || spikes != 0 {
		t.Fatalf("negative step count must fail closed: trace=%v spikes=%d", trace, spikes)
	}
	trace, spikes = SimulateResonateAndFireNeuron(1, math.NaN())
	if trace != nil || spikes != 0 {
		t.Fatalf("non-finite current must fail closed: trace=%v spikes=%d", trace, spikes)
	}
}
