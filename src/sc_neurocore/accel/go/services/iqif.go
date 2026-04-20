// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for iqif

package services

import (
	"math"
)

// IntegerQIFNeuronState holds the neuron state
type IntegerQIFNeuronState struct {
	V float64
	K float64
	VThreshold float64
	VReset float64
	VMin float64
}

// NewIntegerQIFNeuron creates a new IntegerQIFNeuron neuron with default parameters
func NewIntegerQIFNeuron() *IntegerQIFNeuronState {
	return &IntegerQIFNeuronState{
		V: 0.0,
		K: 6.0,
		VThreshold: 1024.0,
		VReset: -1024.0,
		VMin: -2048.0,
	}
}

// Step advances the neuron by one timestep
func (s *IntegerQIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateIntegerQIFNeuron runs the neuron for n steps
func SimulateIntegerQIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewIntegerQIFNeuron()
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
