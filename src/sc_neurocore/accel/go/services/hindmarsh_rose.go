// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hindmarsh_rose

package services

import (
	"math"
)

// HindmarshRoseNeuronState holds the neuron state
type HindmarshRoseNeuronState struct {
	X          float64
	Y          float64
	Z          float64
	B          float64
	R          float64
	S          float64
	XRest      float64
	Dt         float64
	XThreshold float64
}

// NewHindmarshRoseNeuron creates a new HindmarshRoseNeuron neuron with default parameters
func NewHindmarshRoseNeuron() *HindmarshRoseNeuronState {
	return &HindmarshRoseNeuronState{
		X:          -1.6,
		Y:          -10.0,
		Z:          2.0,
		B:          3.0,
		R:          0.001,
		S:          4.0,
		XRest:      -1.6,
		Dt:         0.1,
		XThreshold: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *HindmarshRoseNeuronState) Step(current float64) int {
	if !validateHindmarshRoseState(s) || !finiteHindmarshRose(current) {
		s.X = math.NaN()
		s.Y = math.NaN()
		s.Z = math.NaN()
		return 0
	}
	xPrev := s.X
	dx := s.Y - s.X*s.X*s.X + s.B*s.X*s.X - s.Z + current
	dy := 1.0 - 5.0*s.X*s.X - s.Y
	dz := s.R * (s.S*(s.X-s.XRest) - s.Z)
	s.X += dx * s.Dt
	s.Y += dy * s.Dt
	s.Z += dz * s.Dt
	if !validateHindmarshRoseState(s) {
		s.X = math.NaN()
		s.Y = math.NaN()
		s.Z = math.NaN()
		return 0
	}
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1
	}
	return 0
}

// SimulateHindmarshRoseNeuron runs the neuron for n steps
func SimulateHindmarshRoseNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHindmarshRoseNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

func finiteHindmarshRose(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateHindmarshRoseState(s *HindmarshRoseNeuronState) bool {
	return finiteHindmarshRose(s.X) &&
		finiteHindmarshRose(s.Y) &&
		finiteHindmarshRose(s.Z) &&
		finiteHindmarshRose(s.B) &&
		finiteHindmarshRose(s.R) &&
		finiteHindmarshRose(s.S) &&
		finiteHindmarshRose(s.XRest) &&
		finiteHindmarshRose(s.Dt) &&
		finiteHindmarshRose(s.XThreshold) &&
		s.R > 0.0 &&
		s.S > 0.0 &&
		s.Dt > 0.0
}
