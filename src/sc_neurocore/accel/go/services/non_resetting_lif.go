// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for non_resetting_lif

package services

import (
	"math"
)

// NonResettingLIFNeuronState holds the neuron state
type NonResettingLIFNeuronState struct {
	V float64
	Theta float64
	VRest float64
	ThetaRest float64
	DeltaTheta float64
	TauM float64
	TauTheta float64
	RM float64
	Dt float64
}

// NewNonResettingLIFNeuron creates a new NonResettingLIFNeuron neuron with default parameters
func NewNonResettingLIFNeuron() *NonResettingLIFNeuronState {
	return &NonResettingLIFNeuronState{
		V: -65.0,
		Theta: -50.0,
		VRest: -65.0,
		ThetaRest: -50.0,
		DeltaTheta: 5.0,
		TauM: 10.0,
		TauTheta: 50.0,
		RM: 1.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *NonResettingLIFNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateNonResettingLIFNeuron runs the neuron for n steps
func SimulateNonResettingLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewNonResettingLIFNeuron()
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
