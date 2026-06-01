// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for sigmoid_rate

package services

import (
	"math"
	"testing"
)

func sigmoidRateExactReference(r float64, sigma float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*r + (1.0-decay)*sigma
}

func TestSigmoidRateExactRelaxation(t *testing.T) {
	state := NewSigmoidRateNeuron()
	state.R = 0.25
	state.Tau = 10.0
	state.Beta = 2.0
	state.Theta = 1.0
	state.Dt = 0.5

	sigma, err := sigmoidRateTransfer(state.Beta, 3.0, state.Theta)
	if err != nil {
		t.Fatalf("unexpected transfer error: %v", err)
	}
	expected := sigmoidRateExactReference(state.R, sigma, state.Dt, state.Tau)
	got, err := state.Step(3.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if math.Abs(got-expected) > 1.0e-12 {
		t.Fatalf("rate mismatch: got %.17g want %.17g", got, expected)
	}
}

func TestSigmoidRateLargeTimestepBounded(t *testing.T) {
	state := NewSigmoidRateNeuron()
	state.R = 1.0
	state.Tau = 0.1
	state.Dt = 5.0

	got, err := state.Step(-100.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if got < 0.0 || got > 1.0 {
		t.Fatalf("rate escaped interval: %.17g", got)
	}
	if got >= 1.0e-12 {
		t.Fatalf("rate did not relax close to zero: %.17g", got)
	}
}
