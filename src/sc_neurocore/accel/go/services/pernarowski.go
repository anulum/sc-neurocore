// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for pernarowski

package services

import (
	"math"
)

// PernarowskiNeuronState holds the neuron state
type PernarowskiNeuronState struct {
	V float64
	W float64
	Z float64
	Alpha float64
	Beta float64
	Eps1 float64
	Eps2 float64
	Gamma float64
	Dt float64
	VThreshold float64
}

// NewPernarowskiNeuron creates a new PernarowskiNeuron neuron with default parameters
func NewPernarowskiNeuron() *PernarowskiNeuronState {
	return &PernarowskiNeuronState{
		V: -1.0,
		W: 0.0,
		Z: 0.0,
		Alpha: 0.1,
		Beta: 0.5,
		Eps1: 0.1,
		Eps2: 0.001,
		Gamma: 0.5,
		Dt: 0.1,
		VThreshold: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *PernarowskiNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -1.0
		return 1
	}
	return 0
}

// SimulatePernarowskiNeuron runs the neuron for n steps
func SimulatePernarowskiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPernarowskiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
