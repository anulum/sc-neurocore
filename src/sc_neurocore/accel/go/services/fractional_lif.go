// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fractional_lif

package services

import (
	"math"
)

// FractionalLIFNeuronState holds the neuron state
type FractionalLIFNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	Alpha float64
	Resistance float64
	Dt float64
	MaxHistory float64
}

// NewFractionalLIFNeuron creates a new FractionalLIFNeuron neuron with default parameters
func NewFractionalLIFNeuron() *FractionalLIFNeuronState {
	return &FractionalLIFNeuronState{
		V: 0.0,
		VRest: 0.0,
		VReset: 0.0,
		VThreshold: 1.0,
		Alpha: 0.8,
		Resistance: 1.0,
		Dt: 1.0,
		MaxHistory: 100.0,
	}
}

// Step advances the neuron by one timestep
func (s *FractionalLIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateFractionalLIFNeuron runs the neuron for n steps
func SimulateFractionalLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFractionalLIFNeuron()
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
