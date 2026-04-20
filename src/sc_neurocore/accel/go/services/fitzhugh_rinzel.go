// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_rinzel

package services

import (
	"math"
)

// FitzHughRinzelNeuronState holds the neuron state
type FitzHughRinzelNeuronState struct {
	V float64
	W float64
	Y float64
	A float64
	B float64
	C float64
	D float64
	Delta float64
	Mu float64
	Dt float64
	VThreshold float64
}

// NewFitzHughRinzelNeuron creates a new FitzHughRinzelNeuron neuron with default parameters
func NewFitzHughRinzelNeuron() *FitzHughRinzelNeuronState {
	return &FitzHughRinzelNeuronState{
		V: -1.0,
		W: -0.5,
		Y: 0.0,
		A: 0.7,
		B: 0.8,
		C: -0.775,
		D: 1.0,
		Delta: 0.08,
		Mu: 0.0001,
		Dt: 0.1,
		VThreshold: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *FitzHughRinzelNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -1.0
		return 1
	}
	return 0
}

// SimulateFitzHughRinzelNeuron runs the neuron for n steps
func SimulateFitzHughRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughRinzelNeuron()
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
