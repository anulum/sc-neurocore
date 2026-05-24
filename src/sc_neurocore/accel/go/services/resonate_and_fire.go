// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for resonate_and_fire

package services

import "math"

// ResonateAndFireNeuronState holds the damped oscillator state.
type ResonateAndFireNeuronState struct {
	X         float64
	Y         float64
	B         float64
	Omega     float64
	Threshold float64
	Dt        float64
}

// NewResonateAndFireNeuron creates a new ResonateAndFireNeuron neuron with default parameters.
func NewResonateAndFireNeuron() *ResonateAndFireNeuronState {
	return &ResonateAndFireNeuronState{
		X:         0.0,
		Y:         0.0,
		B:         -0.1,
		Omega:     1.0,
		Threshold: 1.0,
		Dt:        0.05,
	}
}

// Valid reports whether the oscillator parameters are finite and physically admissible.
func (s ResonateAndFireNeuronState) Valid() bool {
	return finite(s.X) &&
		finite(s.Y) &&
		finite(s.B) &&
		finite(s.Omega) && s.Omega > 0.0 &&
		finite(s.Threshold) && s.Threshold > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0
}

// Step advances the resonator by one explicit Euler step. Invalid inputs do not mutate state.
func (s *ResonateAndFireNeuronState) Step(iExt float64) int {
	if !finite(iExt) || !s.Valid() {
		return 0
	}

	dx := (s.B*s.X - s.Omega*s.Y + iExt) * s.Dt
	dy := (s.Omega*s.X + s.B*s.Y) * s.Dt
	nextX := s.X + dx
	nextY := s.Y + dy
	radius := math.Hypot(nextX, nextY)
	if !finite(dx) || !finite(dy) || !finite(nextX) || !finite(nextY) || !finite(radius) {
		return 0
	}

	s.X = nextX
	s.Y = nextY
	if radius >= s.Threshold {
		s.X = 0.0
		s.Y = 0.0
		return 1
	}
	return 0
}

// Reset clears dynamic oscillator state without changing parameters.
func (s *ResonateAndFireNeuronState) Reset() {
	s.X = 0.0
	s.Y = 0.0
}

// SimulateResonateAndFireNeuron runs the neuron for n steps.
func SimulateResonateAndFireNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewResonateAndFireNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = math.Hypot(s.X, s.Y)
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
