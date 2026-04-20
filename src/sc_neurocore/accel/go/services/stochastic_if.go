// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for stochastic_if

package services

import (
	"math"
)

// StochasticIFNeuronState holds the neuron state
type StochasticIFNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	Mu float64
	Sigma float64
	Dt float64
}

// NewStochasticIFNeuron creates a new StochasticIFNeuron neuron with default parameters
func NewStochasticIFNeuron() *StochasticIFNeuronState {
	return &StochasticIFNeuronState{
		V: -70.0,
		VRest: -70.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 20.0,
		Mu: 0.0,
		Sigma: 3.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *StochasticIFNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateStochasticIFNeuron runs the neuron for n steps
func SimulateStochasticIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewStochasticIFNeuron()
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
