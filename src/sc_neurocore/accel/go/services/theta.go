// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for theta

package services

import (
	"errors"
	"math"
)

// ThetaNeuronState holds the neuron state.
type ThetaNeuronState struct {
	Theta float64
	Dt    float64
}

// NewThetaNeuron creates a new ThetaNeuron neuron with default parameters.
func NewThetaNeuron() *ThetaNeuronState {
	return &ThetaNeuronState{
		Theta: 0.0,
		Dt:    0.01,
	}
}

// Valid reports whether the state satisfies the theta phase contract.
func (s ThetaNeuronState) Valid() bool {
	return finite(s.Theta) && finite(s.Dt) && s.Dt > 0.0
}

func wrapTheta(theta float64) float64 {
	return math.Mod(theta+math.Pi, 2.0*math.Pi) - math.Pi
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *ThetaNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) || !s.Valid() {
		return 0, ErrThetaInvalidState
	}

	previous := s.Theta
	cosTheta := math.Cos(s.Theta)
	dtheta := ((1.0 - cosTheta) + (1.0+cosTheta)*iExt) * s.Dt
	nextTheta := s.Theta + dtheta
	if !finite(dtheta) || !finite(nextTheta) {
		return 0, ErrThetaNonFiniteUpdate
	}

	spike := 0
	if previous < math.Pi*0.99 && nextTheta >= math.Pi*0.99 {
		spike = 1
	}
	s.Theta = wrapTheta(nextTheta)
	return spike, nil
}

// Reset restores dynamic state without changing parameters.
func (s *ThetaNeuronState) Reset() {
	s.Theta = 0.0
}

// SimulateThetaNeuron runs the neuron for n steps.
func SimulateThetaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewThetaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.Theta
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrThetaInvalidState    = errors.New("theta state/current must be finite with positive dt")
	ErrThetaNonFiniteUpdate = errors.New("theta phase increment became non-finite")
)
