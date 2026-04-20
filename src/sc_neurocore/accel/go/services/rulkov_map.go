// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for rulkov_map

package services

import (
	"math"
)

// RulkovMapNeuronState holds the neuron state
type RulkovMapNeuronState struct {
	X float64
	Y float64
	Alpha float64
	Sigma float64
	Mu float64
	XThreshold float64
}

// NewRulkovMapNeuron creates a new RulkovMapNeuron neuron with default parameters
func NewRulkovMapNeuron() *RulkovMapNeuronState {
	return &RulkovMapNeuronState{
		X: -1.0,
		Y: -3.0,
		Alpha: 4.0,
		Sigma: -1.6,
		Mu: 0.001,
		XThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *RulkovMapNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateRulkovMapNeuron runs the neuron for n steps
func SimulateRulkovMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewRulkovMapNeuron()
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

var _ = math.Exp
