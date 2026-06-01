// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for resonate_and_fire

package services

import (
	"errors"
	"math"
)

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

// Step advances the resonator by the exact constant-input linear flow. Invalid inputs do not mutate state.
func (s *ResonateAndFireNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) || !s.Valid() {
		return 0, ErrResonateAndFireInvalidState
	}

	nextX, nextY, err := resonateAndFireExactFlow(s.X, s.Y, iExt, s.B, s.Omega, s.Dt)
	if err != nil {
		return 0, err
	}
	radius := math.Hypot(nextX, nextY)
	if !finite(nextX) || !finite(nextY) || !finite(radius) {
		return 0, ErrResonateAndFireNonFiniteUpdate
	}

	if radius >= s.Threshold {
		s.X = 0.0
		s.Y = 0.0
		return 1, nil
	}
	s.X = nextX
	s.Y = nextY
	return 0, nil
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
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = math.Hypot(s.X, s.Y)
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrResonateAndFireInvalidState    = errors.New("resonate-and-fire state/current must be finite and well-formed")
	ErrResonateAndFireNonFiniteUpdate = errors.New("resonate-and-fire exact-flow update became non-finite")
)

func resonateAndFireExactFlow(x float64, y float64, current float64, b float64, omega float64, dt float64) (float64, float64, error) {
	denominator := b*b + omega*omega
	dampingArgument := b * dt
	angle := omega * dt
	if !finite(denominator) || denominator <= 0.0 || !finite(dampingArgument) || !finite(angle) {
		return 0.0, 0.0, ErrResonateAndFireNonFiniteUpdate
	}

	xSS := -b * current / denominator
	ySS := omega * current / denominator
	decay := math.Exp(dampingArgument)
	cosAngle := math.Cos(angle)
	sinAngle := math.Sin(angle)
	if !finite(xSS) || !finite(ySS) || !finite(decay) || !finite(cosAngle) || !finite(sinAngle) {
		return 0.0, 0.0, ErrResonateAndFireNonFiniteUpdate
	}

	dx := x - xSS
	dy := y - ySS
	nextX := xSS + decay*(dx*cosAngle-dy*sinAngle)
	nextY := ySS + decay*(dx*sinAngle+dy*cosAngle)
	if !finite(nextX) || !finite(nextY) {
		return 0.0, 0.0, ErrResonateAndFireNonFiniteUpdate
	}
	return nextX, nextY, nil
}
