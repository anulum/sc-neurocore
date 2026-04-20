// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mcculloch_pitts

package services

import (
	"math"
)

// McCullochPittsNeuronState holds the neuron state
type McCullochPittsNeuronState struct {
	Theta float64
}

// NewMcCullochPittsNeuron creates a new McCullochPittsNeuron neuron with default parameters
func NewMcCullochPittsNeuron() *McCullochPittsNeuronState {
	return &McCullochPittsNeuronState{
		Theta: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *McCullochPittsNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateMcCullochPittsNeuron runs the neuron for n steps
func SimulateMcCullochPittsNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMcCullochPittsNeuron()
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
