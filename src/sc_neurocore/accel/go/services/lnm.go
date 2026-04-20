// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lnm

package services

import (
	"math"
)

// LearnableNeuronModelState holds the neuron state
type LearnableNeuronModelState struct {
	V float64
	Alpha float64
	Beta float64
	Gamma float64
	VThreshold float64
	VReset float64
	FSlope float64
	FShift float64
}

// NewLearnableNeuronModel creates a new LearnableNeuronModel neuron with default parameters
func NewLearnableNeuronModel() *LearnableNeuronModelState {
	return &LearnableNeuronModelState{
		V: 0.0,
		Alpha: 0.9,
		Beta: 0.1,
		Gamma: 0.05,
		VThreshold: 1.0,
		VReset: 0.0,
		FSlope: 5.0,
		FShift: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *LearnableNeuronModelState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateLearnableNeuronModel runs the neuron for n steps
func SimulateLearnableNeuronModel(nSteps int, iExt float64) ([]float64, int) {
	s := NewLearnableNeuronModel()
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
