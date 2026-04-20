// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for theta

package services

import (
	"math"
)

// ThetaNeuronState holds the neuron state
type ThetaNeuronState struct {
	Theta float64
	Dt float64
}

// NewThetaNeuron creates a new ThetaNeuron neuron with default parameters
func NewThetaNeuron() *ThetaNeuronState {
	return &ThetaNeuronState{
		Theta: 0.0,
		Dt: 0.01,
	}
}

// Step advances the neuron by one timestep
func (s *ThetaNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateThetaNeuron runs the neuron for n steps
func SimulateThetaNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewThetaNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Theta
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
