// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for resonate_and_fire service

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
	cosAngle := math.Cos(angle)
	sinAngle := math.Sin(angle)
	return xSS + decay*(dx*cosAngle-dy*sinAngle),
		ySS + decay*(dx*sinAngle+dy*cosAngle)
}

func TestResonateAndFireExactFlowNoSpike(t *testing.T) {
	state := NewResonateAndFireNeuron()
	state.X = 0.3
	state.Y = -0.2
	state.B = -0.2
	state.Omega = 1.7
	state.Threshold = 100.0
	state.Dt = 1.25

	expectedX, expectedY := resonateExactFlow(
		state.X,
		state.Y,
		0.8,
		state.B,
		state.Omega,
		state.Dt,
	)

	spike, err := state.Step(0.8)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("expected no spike, got %d", spike)
	}
	if math.Abs(state.X-expectedX) > 1.0e-12 || math.Abs(state.Y-expectedY) > 1.0e-12 {
		t.Fatalf(
			"exact-flow mismatch: got (%0.17g, %0.17g), want (%0.17g, %0.17g)",
			state.X,
			state.Y,
			expectedX,
			expectedY,
		)
	}
}

func TestResonateAndFireInvalidStatePreservesState(t *testing.T) {
	state := NewResonateAndFireNeuron()
	state.X = 0.25
	state.Y = -0.5
	beforeX := state.X
	beforeY := state.Y
	state.Dt = 0.0

	if _, err := state.Step(0.5); err == nil {
		t.Fatal("expected invalid-state error")
	}
	if state.X != beforeX || state.Y != beforeY {
		t.Fatalf("state mutated on invalid input: got (%v, %v)", state.X, state.Y)
	}
}
