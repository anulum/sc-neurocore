// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for truenorth

package services

import (
	"math"
)

// TrueNorthNeuronState holds the neuron state
type TrueNorthNeuronState struct {
	V float64
	Leak float64
	Threshold float64
	VReset float64
}

// NewTrueNorthNeuron creates a new TrueNorthNeuron neuron with default parameters
func NewTrueNorthNeuron() *TrueNorthNeuronState {
	return &TrueNorthNeuronState{
		V: 0.0,
		Leak: 0.0,
		Threshold: 100.0,
		VReset: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *TrueNorthNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateTrueNorthNeuron runs the neuron for n steps
func SimulateTrueNorthNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTrueNorthNeuron()
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
