// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for spinnaker2

package services

import (
	"math"
)

// SpiNNaker2NeuronState holds the neuron state
type SpiNNaker2NeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	DecayMult float64
	DecayShift float64
	RefracSteps float64
	RefracCount float64
}

// NewSpiNNaker2Neuron creates a new SpiNNaker2Neuron neuron with default parameters
func NewSpiNNaker2Neuron() *SpiNNaker2NeuronState {
	return &SpiNNaker2NeuronState{
		V: 0.0,
		VRest: 0.0,
		VReset: 0.0,
		VThreshold: 1024.0,
		DecayMult: 243.0,
		DecayShift: 8.0,
		RefracSteps: 2.0,
		RefracCount: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *SpiNNaker2NeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateSpiNNaker2Neuron runs the neuron for n steps
func SimulateSpiNNaker2Neuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewSpiNNaker2Neuron()
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
