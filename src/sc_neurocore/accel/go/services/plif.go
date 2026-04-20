// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for plif

package services

import (
	"math"
)

// ParametricLIFNeuronState holds the neuron state
type ParametricLIFNeuronState struct {
	V float64
	A float64
	Threshold float64
	Dt float64
}

// NewParametricLIFNeuron creates a new ParametricLIFNeuron neuron with default parameters
func NewParametricLIFNeuron() *ParametricLIFNeuronState {
	return &ParametricLIFNeuronState{
		V: 0.0,
		A: 0.0,
		Threshold: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *ParametricLIFNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateParametricLIFNeuron runs the neuron for n steps
func SimulateParametricLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewParametricLIFNeuron()
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
